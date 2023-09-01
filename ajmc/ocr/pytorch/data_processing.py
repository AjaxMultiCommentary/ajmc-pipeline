import random
import time
from pathlib import Path
from typing import List, Dict

import math
import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from ajmc.commons.unicode_utils import harmonise_unicode
from ajmc.ocr import variables as ocr_vs


class OcrLine:
    """A custom class for OCR lines.

    Note:
        - A line corresponds both to a txt and an img if in train mode, only an image if not.
        - img : the image of the line
            - img_width: The width of the original image
            - img_height: The height of the original image
            - img_padding_length: The length of padding added for batching in the decoder
        - Chunks : the chunks of the image of the line
            - chunks_width: The widths of chunks
            - chunks_height = img height
            - chunks_count: The number of chunks
            - last_chunk_padding: The length of the padding added to the last chunk.
        - text : the text corresponding to the image.
            - text_length: The length of the text

    """

    # Todo test this for speed
    def __init__(self,
                 img_path: Path,
                 img_height: int,
                 chunk_width: int,
                 chunk_overlap: int,
                 classes_to_indices: dict
                 ):
        start_time = time.time()
        img_tensor = prepare_img(img_path, img_height=img_height)
        self.img_width = img_tensor.shape[2]
        self.chunks = chunk_img_tensor(img_tensor, chunk_width, chunk_overlap)

        text = img_path.with_suffix(ocr_vs.GT_TEXT_EXTENSION).read_text(encoding='utf-8')  # Todo change store this ??
        self.text = harmonise_unicode(text)
        self.text_tensor = torch.tensor([classes_to_indices[c] for c in text])


class OcrBatch:

    def __init__(self, ocr_lines: List[OcrLine]):
        self.ocr_lines = ocr_lines

    @property
    def img_widths(self) -> tuple:
        return tuple(l.img_width for l in self.ocr_lines)

    @property
    def chunks(self) -> torch.Tensor:
        return torch.cat([l.chunks for l in self.ocr_lines], dim=0)

    @property
    def chunks_to_img_mapping(self) -> List[int]:
        return [len(l.chunks) for l in self.ocr_lines]

    @property
    def text_lengths(self) -> tuple:
        return tuple(l.text_tensor.shape[0] for l in self.ocr_lines)

    @property
    def texts_tensor(self) -> torch.Tensor:
        return pad_sequence([l.text_tensor for l in self.ocr_lines], batch_first=True)

    @property
    def texts(self) -> List[str]:
        return [l.text for l in self.ocr_lines]


def invert_image_tensor(image_tensor):
    return 255 - image_tensor


def normalize_image_tensor(image_tensor) -> torch.Tensor:
    """Prepares an image tensor for training."""
    return image_tensor.float() / 255


def crop_image_tensor_to_nonzero(image_tensor: torch.Tensor) -> torch.Tensor:
    """Crops an image tensor of shape (1, height, width) to the smallest bounding box containing all non-zero pixels."""

    non_zero = torch.nonzero(image_tensor)

    xmin = non_zero[:, 1].min()
    xmax = non_zero[:, 1].max()
    ymin = non_zero[:, 2].min()
    ymax = non_zero[:, 2].max()

    return image_tensor[:, xmin:xmax + 1, ymin:ymax + 1]


def compute_n_chunks(img_tensor,
                     chunk_width: int,
                     chunk_overlap: int):
    """Computes the number a chunks that can be extracted from an image tensor, given a chunk width and an overlap.

    Note:
        We have $n * w - (n - 1) * o = W+padding$ where $n$ is the number of chunks, $w$ is the chunk width, $o$ is the overlap
        and $W$ is the image width (without padding), which can be rewritten to : $n (w - o) + o = W+padding$. Therefore,
        $n = (W+padding - o) / (w - o)$, and $n = (W+padding - o) // (w - o) (+ 1 if there is a remainder)$.

    Args:
        img_tensor: The image tensor from which to extract the chunks.
        chunk_width: The width of the chunks.
        chunk_overlap: The overlap between the chunks.

    Returns:
        The number of chunks that can be extracted from the image tensor.
    """
    remainder = (img_tensor.shape[2] - chunk_overlap) % (chunk_width - chunk_overlap)

    return (img_tensor.shape[2] - chunk_overlap) // (chunk_width - chunk_overlap) + (1 if remainder else 0)


def compute_padding(img_tensor,
                    n_chunks: int,
                    chunk_width: int,
                    chunk_overlap: int):
    """Computes the padding to apply to an image tensor to extract chunks of a given width with a given overlap.

    Args:
        img_tensor: The image tensor from which to extract the chunks.
        n_chunks: The number of chunks to extract.
        chunk_width: The width of the chunks.
        chunk_overlap: The overlap between the chunks.

    Returns:
        The padding to apply to the image tensor.
    """

    return n_chunks * chunk_width - (n_chunks - 1) * chunk_overlap - img_tensor.shape[2]


def chunk_img_tensor(img_tensor,
                     chunk_width: int,
                     chunk_overlap: int) -> torch.Tensor:
    n_chunks = compute_n_chunks(img_tensor, chunk_width, chunk_overlap)
    padding = compute_padding(img_tensor, n_chunks, chunk_width, chunk_overlap)

    # Pad the tensor
    img_tensor = torch.nn.functional.pad(img_tensor, (0, padding), mode='constant', value=0)

    # Chunk the tensor
    chunks = []
    for i in range(n_chunks):
        chunks.append(img_tensor[:, :, i * (chunk_width - chunk_overlap):i * (chunk_width - chunk_overlap) + chunk_width])

    return torch.stack(chunks, dim=0)


def prepare_img(img_path: Path,
                img_height: int) -> torch.Tensor:
    """Prepares an image tensor for training."""

    img_tensor = read_image(str(img_path), mode=ImageReadMode.GRAY)
    img_tensor = invert_image_tensor(img_tensor)
    img_tensor = transforms.Resize(img_height, antialias=True)(img_tensor)
    img_tensor = crop_image_tensor_to_nonzero(img_tensor)
    img_tensor = normalize_image_tensor(img_tensor)

    return img_tensor


def recompose_chunks(chunks: torch.Tensor,
                     chunk_overlap: int) -> torch.Tensor:
    """Reassembles chunks of an image tensor into a single image tensor.

    Args:
        chunks: The chunks to reassemble, in shape (n_chunks, width, height). Note that the channels dimension is not
            present, as the chunks are grayscale images. processed by the encoder.
        chunk_overlap: The overlap between the chunks.

    Returns:
        The reassembled image tensor, in shape (width, height).
    """

    reassembled_img = chunks[0][:-int(chunk_overlap / 2), :]  # take the first chunk, remove the overlap
    for i in range(1, len(chunks)):
        if i == len(chunks) - 1:  # for the last chunk, we dont cut the end overlap as there is none
            reassembled_img = torch.cat([reassembled_img, chunks[i][int(chunk_overlap / 2):, :]], dim=0)
        else:  # for the other chunks, we cut the begin and end overlap
            reassembled_img = torch.cat([reassembled_img, chunks[i][int(chunk_overlap / 2):-int(chunk_overlap / 2), :]], dim=0)

    # Todo 👁️ the end of this tensor could be chunked to non-zero values to recut the introduced padding
    # This will have to be done somehow, imgs offsets requires it.
    return reassembled_img


# Todo see if works with gpu:
def recompose_batched_chunks(batched_chunks: torch.Tensor, mapping: List[int], chunk_overlap: int) -> torch.Tensor:
    """Apply `recompose_chunks` to a batch of chunks.

    Args:
        batched_chunks: The batch of chunks, in shape (n_chunks, chunk_width, chunk_height [or num_classes if performed after decoder).
        mapping: The mapping from the batch to the original image tensor, in shape (1). The mapping is a tensor of
            integers, where each integer represents the number of chunks that were extracted from the original image
            tensor. Eg : [3, 2] means that the first image tensor was chunked into 3 chunks, the second into 2 chunks.
        chunk_overlap: The overlap between the chunks.

    Returns:
        The reassembled image tensor, in shape (n_images, 1, img_height, img_width+eventual padding).
    """
    chunk_groups = torch.split(batched_chunks, mapping, dim=0)
    reassembled = [recompose_chunks(group, chunk_overlap) for group in chunk_groups]
    return pad_sequence(reassembled, batch_first=True)


class OcrIterDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 data_dir: Path,
                 classes: str,
                 max_batch_size: int,
                 img_height: int,
                 chunk_width: int,
                 chunk_overlap: int,
                 classes_to_indices: Dict[str, int],
                 indices_to_classes: Dict[int, str],
                 is_training: bool = True,
                 shuffle: bool = True,
                 per_worker_steps_run: int = 0):
        self.data_dir = data_dir
        self.classes = classes
        self.max_batch_size = max_batch_size
        self.img_height = img_height
        self.chunk_width = chunk_width
        self.chunk_overlap = chunk_overlap
        self.classes_to_indices = classes_to_indices
        self.indices_to_classes = indices_to_classes
        self.img_paths = list(self.data_dir.rglob('*' + ocr_vs.IMG_EXTENSION))
        if shuffle:
            random.shuffle(self.img_paths)

        self.is_training = is_training  # Todo see how we handle this
        self.per_worker_steps_run = per_worker_steps_run


    def distribute(self):
        """Distributes the datasets accross workers, see https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset"""
        worker_start = self.per_worker_steps_run
        worker_end = len(self.img_paths)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            samples_per_worker = int(math.ceil(worker_end / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_start = worker_id * samples_per_worker + self.per_worker_steps_run
            worker_end = min(worker_start + samples_per_worker - self.per_worker_steps_run, worker_end)

        return worker_start, worker_end

    def _custom_iterator(self):
        """
        We keep tensors in the following form:
            id_bs0.pt
            id_bs0.txt

        """
        start, end = self.distribute()
        next_index = start

        # Create the first line
        ocr_line = OcrLine(self.img_paths[next_index],
                           img_height=self.img_height,
                           chunk_overlap=self.chunk_overlap,
                           chunk_width=self.chunk_width,
                           classes_to_indices=self.classes_to_indices)

        while next_index < end:

            if ocr_line.chunks.shape[0] > self.max_batch_size:  # If there are more chunks that batch_size, skip
                next_index += 1
                if next_index < end:
                    ocr_line = OcrLine(self.img_paths[next_index],
                                       img_height=self.img_height,
                                       chunk_overlap=self.chunk_overlap,
                                       chunk_width=self.chunk_width,
                                       classes_to_indices=self.classes_to_indices)
                    continue
                else:
                    break

            chunks_count: int = ocr_line.chunks.shape[0]
            ocr_lines = [ocr_line]
            next_index += 1

            # we get the number of chunks (=shape[0]) in the next tensor

            while next_index < end:
                ocr_line = OcrLine(self.img_paths[next_index],
                                   img_height=self.img_height,
                                   chunk_overlap=self.chunk_overlap,
                                   chunk_width=self.chunk_width,
                                   classes_to_indices=self.classes_to_indices)
                if ocr_line.chunks.shape[0] + chunks_count > self.max_batch_size:
                    break
                else:
                    ocr_lines.append(ocr_line)
                    chunks_count += ocr_line.chunks.shape[0]
                    next_index += 1

            yield OcrBatch(ocr_lines)  # Will this have to be converted to tuple ?

    def __iter__(self):
        return self._custom_iterator()


def get_custom_dataloader(train_dataset: torch.utils.data.Dataset,
                          num_workers: int) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(train_dataset,
                                       batch_size=None,
                                       batch_sampler=None,
                                       num_workers=num_workers,
                                       collate_fn=lambda x: x, )
