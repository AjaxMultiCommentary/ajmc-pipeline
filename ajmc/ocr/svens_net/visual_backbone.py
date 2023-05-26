import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torchvision.models.densenet import _Transition, _DenseBlock
from collections import OrderedDict
from typing import Iterable

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CustomDenseNet(nn.Module):

    def __init__(self,
                 growth_rate: int = 32,
                 block_config: Iterable[int] = (6, 12, 24, 16),
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 1000,
                 memory_efficient: bool = False, ) -> None:
        super().__init__()

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

        # Linear layer
        # Here remove the linear layer
        # self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = torch.flatten(out, 1)

        return out


backbone = CustomDenseNet(growth_rate=12,
                          block_config=[24, 24, 24],
                          num_init_features=64)

#%% get the image
from PIL import Image

from ajmc.commons import variables as vs


img_path = vs.COMMS_DATA_DIR / 'sophoclesplaysa05campgoog' / 'ocr/gt_file_pairs/sophoclesplaysa05campgoog_0177_22.png'
img = Image.open(img_path, mode='r')
img = img.convert('L')
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img).unsqueeze(0)

backbone(img_tensor).shape

"Schon Homer Il. 21, 50 γυμνόν," == 'Schon Homer Il. 21, 50 γυμνόν,'

#%%

from pathlib import Path

dir = Path('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/cu31924087948174/ocr/gt_file_pairs')
for path in dir.glob('*.txt'):
    txt = path.read_text(encoding='utf-8')
    if txt.startswith('φευκτὰν C3.'):
        print(path)
        break
