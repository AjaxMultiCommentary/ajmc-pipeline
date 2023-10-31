"""Model classes for torch based ocr models."""

from collections import OrderedDict
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import _Transition, _DenseBlock

from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.ocr.pytorch.ctc_decoder_torch import GreedyDecoder
from ajmc.ocr.pytorch.data_processing import recompose_batched_chunks

logger = get_ajmc_logger(__name__)


class OcrTorchModel(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        self.classes = config['classes']
        self.indices_to_classes = config['indices_to_classes']
        self.chunk_overlap = config['chunk_overlap']

        if config.get('densenetbackbone', False):
            self.backbone = DenseNetBackbone(chunk_width=config['chunk_width'],
                                             encoder_d_model=config['encoder']["TransformerEncoderLayer"]["d_model"],
                                             **config['densenetbackbone'])
        else:
            self.backbone = self.reshape_input
            logger.warning('No backbone implemented')

        self.encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(**config['encoder']["TransformerEncoderLayer"]),
                                             **config['encoder']["TransformerEncoder"])

        self.decoder = nn.Linear(**config['decoder'])

        self.ctc_decoder = GreedyDecoder(labels=self.classes, blank_index=0)


    def reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        return torch.reshape(x, (x.shape[0], x.shape[3], x.shape[2]))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A torch tensor of shape (batch_size, 1, height, width) representing a grayscale image chunks, normalized to [0, 1] and white on black.
            mapping: A torch tensor of shape (1) representing the mapping from the image chunks to the original image.
        """

        logger.debug(f'input shape: {x.shape}')
        x = self.backbone(x)
        logger.debug(f'backbone output shape: {x.shape}')
        x = self.encoder(x)
        logger.debug(f'encoder output shape: {x.shape}')
        x = self.decoder(x)
        logger.debug(f'decoder output shape: {x.shape}')

        return x

    def predict(self, x, chunks_to_img_mapping) -> List[str]:
        """Predicts the text in a batch of image tensors.

        Args:
            x: A torch tensor of shape (batch_size, width, height).
            chunks_to_img_mapping:

        Returns:
            A list of strings representing the predicted text.
        """
        with torch.no_grad():
            outputs = self.forward(x)

        outputs = recompose_batched_chunks(outputs,
                                           mapping=chunks_to_img_mapping,
                                           chunk_overlap=self.chunk_overlap)

        outputs = torch.nn.functional.log_softmax(outputs, dim=2)

        strings, offsets = self.ctc_decoder.decode(outputs)

        return strings


class DenseNetBackbone(nn.Module):
    """A customised DenseNet backbone.
    
    Customisation notably including single channel inputs, pooling and squeezing the outputs, as
    well as adapting accepted block configurations.
    """

    def __init__(self,
                 chunk_width: int,
                 encoder_d_model: int,
                 growth_rate: int = 32,
                 block_config: Iterable[int] = (6, 12, 24, 16),
                 num_init_features: int = 32,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 memory_efficient: bool = False) -> None:
        super().__init__()
        self.chunk_width = chunk_width
        self.encoder_d_model = encoder_d_model

        # First convolution
        self.features = nn.Sequential(
                OrderedDict(
                        [
                            ("conv0", nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                            # Here we changed the number of input channel to 1, so as to process grayscale images
                            ("norm0", nn.BatchNorm2d(num_init_features)),
                            ("relu0", nn.ReLU(inplace=True)),
                            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                        ]
                )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                    num_layers=num_layers,
                    num_input_features=num_features,
                    bn_size=bn_size,
                    growth_rate=growth_rate,
                    drop_rate=drop_rate,
                    memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # for i, feature in enumerate(self.features):
        #     # logger.debug(f'Shape at feature {i}: {x.shape}')
        #     x = feature(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[3])
        x = F.adaptive_avg_pool2d(x, (self.chunk_width, self.encoder_d_model))
        x = F.relu(x, inplace=True)

        return x
