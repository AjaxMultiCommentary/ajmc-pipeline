import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Generator

import math
import torch
import unicodedata
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.commons.unicode_utils import harmonise_unicode
from ajmc.ocr import variables as ocr_vs

logger = get_ajmc_logger(__name__)


# Todo add a training mode where the text is not given

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

    def __init__(self,
                 img_path: Path,
                 img_height: int,
                 chunk_width: int,
                 chunk_overlap: int,
                 classes_to_indices: dict
                 ):
        img_tensor = prepare_img(img_path, img_height=img_height)
        self.img_width = img_tensor.shape[2]
        self.chunks = chunk_img_tensor(img_tensor, chunk_width, chunk_overlap)
        self.img_path = img_path

        text = img_path.with_suffix(ocr_vs.GT_TEXT_EXTENSION).read_text(encoding='utf-8')
        self.text = unicodedata.normalize('NFD', harmonise_unicode(text))
        self.text_tensor = torch.tensor([classes_to_indices[c] for c in self.text])


class OcrBatch:

    def __init__(self, ocr_lines: List[OcrLine]):
        self.ocr_lines = ocr_lines
        self.img_widths = tuple(l.img_width for l in self.ocr_lines)
        self.chunks = torch.cat([l.chunks for l in self.ocr_lines], dim=0)
        self.chunks_to_img_mapping = [len(l.chunks) for l in self.ocr_lines]
        self.text_lengths = tuple(l.text_tensor.shape[0] for l in self.ocr_lines)
        self.texts_tensor = pad_sequence([l.text_tensor for l in self.ocr_lines], batch_first=True)
        self.texts = tuple(l.text for l in self.ocr_lines)


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
        We have :math:`n * w - (n - 1) * o = W + padding` where :math:`n` is the number of chunks, :math:`w` is the chunk width, :math:`o` is the overlap
        and :math:`W` is the image width (without padding), which can be rewritten to : :math:`n (w - o) + o = W + padding`. Therefore,
        :math:`n = (W+padding - o) / (w - o)`, and :math:`n = (W+padding - o) // (w - o)` (:math:`+ 1` if there is a remainder).

    Args:
        img_tensor: The image tensor from which to extract the chunks.
        chunk_width: The width of the chunks.
        chunk_overlap: The overlap between the chunks.

    Returns:
        The number of chunks that can be extracted from the image tensor.
    """

    # We use max to avoid negative numbers for very tiny images.
    diff_img_width_overlap = max(img_tensor.shape[2] - chunk_overlap, 1)

    remainder = diff_img_width_overlap % (chunk_width - chunk_overlap)

    return diff_img_width_overlap // (chunk_width - chunk_overlap) + (1 if remainder else 0)


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
    """Chunks an image tensor of shape (1, height, width) into chunks of shape (N_chunks, height, chunk_width).

    Args:
        img_tensor: The image tensor to chunk, of shape (1, height, width).
        chunk_width: The desired width of the chunks.
        chunk_overlap: The desired overlap between the chunks.

    Returns:
        The image tensor chunked into chunks of shape (N_chunks, 1,  height, chunk_width).
    """

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
    """Prepares an image tensor for training.

    Args:
        img_path: The path to a grayscale image of shape (1, initial_height, initial_width).
        img_height: The height to which to resize the image.

    Returns:
        The prepared image tensor, in shape (1, img_height, resized_width).
    """

    img_tensor = read_image(str(img_path), mode=ImageReadMode.GRAY)
    img_tensor = invert_image_tensor(img_tensor)
    img_tensor = crop_image_tensor_to_nonzero(img_tensor)

    # We calculate the entire size to be able to cope with images longer than large
    # See docs on resize (https://pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html)
    resized_size = (img_height, int((img_height / img_tensor.shape[1]) * img_tensor.shape[2]))
    img_tensor = transforms.Resize(resized_size, antialias=True)(img_tensor)
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
        The reassembled tensor, in shape (width, height).
    """

    # take the first chunk, remove the overlap
    reassembled = chunks[0][:-int(chunk_overlap / 2), :].squeeze(0)
    for i in range(1, len(chunks)):
        if i == len(chunks) - 1:  # for the last chunk, we dont cut the end overlap as there is none
            reassembled = torch.cat([reassembled, chunks[i][int(chunk_overlap / 2):, :]], dim=0)
        else:  # for the other chunks, we cut the begin and end overlap
            reassembled = torch.cat([reassembled, chunks[i][int(chunk_overlap / 2):-int(chunk_overlap / 2), :]], dim=0)

    # Todo 👁️ the end of this tensor could be chunked to non-zero values to recut the introduced padding
    # This will have to be done somehow, imgs offsets requires it.
    return reassembled


def recompose_batched_chunks(batched_chunks: torch.Tensor, mapping: List[int], chunk_overlap: int) -> torch.Tensor:
    """Apply ``recompose_chunks`` to a batch of chunks.

    Warning:
        This function is to be applied on a processed batch of chunks, with the shape
        (n_chunks, chunk_width, chunk_height [or num_classes if performed after decoder).

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
    """Dataset for OCR training.

    Warning:
        ``OcrIterDataset`` is an infinite iterable dataset, and as such, it does not have a ``__len__`` method. This means that:

            * ``OcrIterDataset`` it cannot be used with a ``torch.utils.data.DataLoader`` with ``batch_size > 1``. Actually, \
            ``OcrIterDataset.__iter__()`` already returns batches of size inferior to ``max_batch_size`` at each iteration.
            * ``OcrIterDataset`` is infinite: Use it with ``next(iter())`` in a ``for`` loop, or with a defined range.
            * ``OcrIterDataset.date_len`` is **not** the length of the dataset, but the length of the list of images the dataset will infinitely iterate on. It is therefore not the same as ``__len__``.


    Args:
        data_dir: The directory containing the images to train on.
        classes: The classes to train on.
        max_batch_size: The maximum batch size to return at each iteration.
        img_height: The height to which to resize the images.
        chunk_width: The width of the chunks to extract from the images.
        chunk_overlap: The overlap between the chunks.
        classes_to_indices: A mapping from the classes to their indices.
        indices_to_classes: A mapping from the indices to their classes.
        is_training: Whether the dataset is used for training or validation.
        shuffle: Whether to shuffle the dataset.
        per_worker_steps_run: The number of steps already run by each worker. This is used to compute the number of chunks to
            skip at the beginning of the dataset, so that each worker starts at a different point in the dataset.

    """

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
                 num_workers: int = 1,
                 per_worker_steps_run: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.classes = classes
        self.max_batch_size = max_batch_size
        self.img_height = img_height
        self.chunk_width = chunk_width
        self.chunk_overlap = chunk_overlap
        self.classes_to_indices = classes_to_indices
        self.indices_to_classes = indices_to_classes
        self.is_training = is_training  # Todo see how we handle this
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.per_worker_steps_run = per_worker_steps_run

        # Get the paths of the images
        self.img_paths = list(self.data_dir.rglob('*' + ocr_vs.IMG_EXTENSION))
        if self.shuffle:
            random.shuffle(self.img_paths)

        # Distribute the dataset accross workers
        self.worker_id = int(os.environ.get('RANK', 0))
        self.data_len = len(self.img_paths)
        self.files_generator = self.yield_files(*self.distribute())
        self.batch_generator = self.yield_batches()

    def distribute(self) -> Tuple[int, int, int]:
        """Distributes the datasets accross workers.

        Note:
            See https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for more information.

        Returns:
            The default start, re-start (which corresponds to the defaults + the number of steps already run) and end indices for the current worker.
        """

        if self.num_workers == 1:  # single-process data loading, return the full iterator
            worker_default_start = 0
            worker_restart = self.per_worker_steps_run
            worker_end = self.data_len

        else:  # Multi-process data loading, split the dataset
            logger.info(f'Distributing data across {self.num_workers} workers')
            worker_id = self.worker_id
            samples_per_worker = int(math.ceil(self.data_len / float(self.num_workers)))
            worker_default_start = worker_id * samples_per_worker
            worker_restart = worker_default_start + self.per_worker_steps_run
            worker_end = min(worker_default_start + samples_per_worker, self.data_len)

            logger.info(f'Worker {self.worker_id} is starting at step {worker_restart}')

        return worker_default_start, worker_restart, worker_end

    def yield_files(self, default_start: int, restart: int, end: int) -> Generator[Path, None, None]:
        """Yields files infinitely, starting at the given start index.

        Note:
            For the first iteration, the start index is the worker's actual re-start index. For the following iterations,
            the start index is the default start index (ie the start index for the first iteration). If ``restart == default_start`` then exactly
            the same happens.

        Args:
            default_start: The default start index.
            restart: The worker's actual re-start index (in case of restarting from checkpoint).
            end: The end index.

        Yields:
            The files in the dataset, starting at the given start index.
        """
        for i in range(restart, end):  # We start at the worker's actual re-start index
            logger.debug(f'Worker {self.worker_id} Yielding file {self.img_paths[i].stem}')
            yield self.img_paths[i]

        while True:
            for i in range(default_start, end):
                logger.debug(f'Worker {self.worker_id} Yielding file {self.img_paths[i].stem}')
                yield self.img_paths[i]

    def yield_batches(self) -> Generator[OcrBatch, None, None]:
        """Yields batches of chunks infinitely.

        # Todo : docs
        """

        # Create the first line
        ocr_line = OcrLine(next(self.files_generator),
                           img_height=self.img_height,
                           chunk_overlap=self.chunk_overlap,
                           chunk_width=self.chunk_width,
                           classes_to_indices=self.classes_to_indices)

        while True:

            if ocr_line.chunks.shape[0] > self.max_batch_size:  # If there are more chunks that batch_size, skip the line
                ocr_line = OcrLine(next(self.files_generator),
                                   img_height=self.img_height,
                                   chunk_overlap=self.chunk_overlap,
                                   chunk_width=self.chunk_width,
                                   classes_to_indices=self.classes_to_indices)
                continue

            chunks_count: int = ocr_line.chunks.shape[0]
            ocr_lines = [ocr_line]

            while True:
                ocr_line = OcrLine(next(self.files_generator),
                                   img_height=self.img_height,
                                   chunk_overlap=self.chunk_overlap,
                                   chunk_width=self.chunk_width,
                                   classes_to_indices=self.classes_to_indices)

                if ocr_line.chunks.shape[0] + chunks_count > self.max_batch_size:
                    break

                ocr_lines.append(ocr_line)
                chunks_count += ocr_line.chunks.shape[0]

            yield OcrBatch(ocr_lines)

    def __iter__(self):
        return self.batch_generator


def get_custom_dataloader(train_dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(train_dataset,
                                       batch_size=None,
                                       batch_sampler=None,
                                       num_workers=0,
                                       collate_fn=lambda x: x, )
