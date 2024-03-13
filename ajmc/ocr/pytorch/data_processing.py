# TODO : revise the docs
import os
import random
import unicodedata
from abc import abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Generator, Optional, Any, Callable, Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.commons.unicode_utils import harmonise_unicode
from ajmc.ocr import variables as ocr_vs


logger = get_ajmc_logger(__name__)


class TorchTrainingLine:
    """A custom class for OCR training lines

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

    # #@profile
    def __init__(self,
                 img_path: Path,
                 img_height: int,
                 chunk_width: int,
                 chunk_overlap: int,
                 classes_to_indices: Dict[str, int],
                 special_mapping: Dict[str, str],
                 unknown_char_index: int = 1):
        img_tensor = read_image(str(img_path), mode=ImageReadMode.GRAY).requires_grad_(False)
        img_tensor = prepare_img_tensor(img_tensor, img_height=img_height)
        self.img_width: int = img_tensor.shape[2]
        self.chunks = chunk_img_tensor(img_tensor, chunk_width, chunk_overlap)
        self.text = prepare_text(img_path.with_suffix(ocr_vs.GT_TEXT_EXTENSION).read_text(encoding='utf-8'), special_mapping=special_mapping)
        self.text_tensor = torch.tensor([classes_to_indices.get(c, unknown_char_index) for c in self.text])


class TorchInferenceLine:
    """A custom class for torch lines at inference time, were the text is not given and where the line image
    is provided directly as a ``torch.Tensor``.

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

    # #@profile
    def __init__(self,
                 line_id: str,
                 img_tensor: torch.Tensor,
                 img_height: int,
                 chunk_width: int,
                 chunk_overlap: int):
        """Initializes a ``TorchInferenceLine`` instance.

        Note:
            This class is different from the ``TorchTrainingLine``, notably because:
                - The text is not given, as it is to be inferred.
                - The image is given as a ``torch.Tensor`` and not as a file path.

        Args:
            line_id (str): The id of the line (e.g. ``{page_id}_{x1}_{y1}_{x2}_{y2}``)
            img_tensor (torch.Tensor): The image tensor of the line, in shape (1, line_height, line_width),
            as read by ``read_image(str(img_tensor), mode=ImageReadMode.GRAY).requires_grad_(False)``
            img_height (int): The desired resizing height of the image (a model parameter)
            chunk_width (int): The width of the chunks to be created from the image
            chunk_overlap (int): The overlap between the chunks to be created from the image
        """

        self.line_id = line_id
        self.img_tensor = prepare_img_tensor(img_tensor, img_height=img_height)
        self.img_width: int = self.img_tensor.shape[2]
        self.chunks = chunk_img_tensor(self.img_tensor, chunk_width, chunk_overlap)


class TorchTrainingBatch:

    # #@profile
    def __init__(self,
                 img_widths: Tuple[int],
                 chunks: torch.Tensor,
                 chunks_to_img_mapping: List[int],
                 text_lengths: Tuple[Any],
                 texts_tensor: torch.Tensor,
                 texts: Tuple[str],
                 ):
        self.img_widths = img_widths
        self.chunks = chunks
        self.chunks_to_img_mapping = chunks_to_img_mapping
        self.text_lengths = text_lengths
        self.texts_tensor = texts_tensor
        self.texts = texts

    @classmethod
    def from_lines(cls, ocr_lines: List[TorchTrainingLine]):
        return cls(img_widths=tuple(l.img_width for l in ocr_lines),
                   chunks=torch.cat([l.chunks for l in ocr_lines], dim=0),
                   chunks_to_img_mapping=[len(l.chunks) for l in ocr_lines],
                   text_lengths=tuple(l.text_tensor.shape[0] for l in ocr_lines),
                   texts_tensor=pad_sequence([l.text_tensor for l in ocr_lines], batch_first=True),
                   texts=tuple(l.text for l in ocr_lines))

    def to_dict(self):
        return {'img_widths': self.img_widths,
                'chunks': self.chunks,
                'chunks_to_img_mapping': self.chunks_to_img_mapping,
                'text_lengths': self.text_lengths,
                'texts_tensor': self.texts_tensor,
                'texts': self.texts}

    def change_texts(self, chars_to_special_classes: Dict[str, str], classes_to_indices: Dict[str, int]):
        self.texts = tuple(prepare_text(t, chars_to_special_classes) for t in self.texts)
        self.texts_tensor = pad_sequence([torch.tensor([classes_to_indices.get(c, 1) for c in t]) for t in self.texts], batch_first=True)
        self.text_lengths = tuple(len(t) for t in self.texts)


class TorchInferenceBatch:

    # #@profile
    def __init__(self,
                 line_ids: List[str],
                 img_widths: Tuple[int],
                 chunks: torch.Tensor,
                 chunks_to_img_mapping: List[int],
                 ):
        self.line_ids = line_ids
        self.img_widths = img_widths
        self.chunks = chunks
        self.chunks_to_img_mapping = chunks_to_img_mapping

    @classmethod
    def from_lines(cls, ocr_lines: List[TorchInferenceLine]):
        return cls(line_ids=[l.line_id for l in ocr_lines],
                   img_widths=tuple(l.img_width for l in ocr_lines),
                   chunks=torch.cat([l.chunks for l in ocr_lines], dim=0),
                   chunks_to_img_mapping=[len(l.chunks) for l in ocr_lines])


class TorchOcrDataset(torch.utils.data.Dataset):

    def __init__(self,
                 max_batch_size: int,
                 img_height: int,
                 chunk_width: int,
                 chunk_overlap: int,
                 data_dir: Optional[Path] = None,
                 img_paths: Optional[List[Path]] = None,
                 num_workers: int = 1,
                 per_worker_steps_run: int = 0):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.img_height = img_height
        self.chunk_width = chunk_width
        self.chunk_overlap = chunk_overlap
        self.per_worker_steps_run = per_worker_steps_run

        # Get the paths of the images
        if img_paths is not None:
            logger.info(f'Using {len(img_paths)} images from given list of paths.')
            self.img_paths = img_paths

        else:
            self.img_paths = sorted(data_dir.rglob('*' + ocr_vs.IMG_EXTENSION), key=lambda x: x.stem)
            logger.info(f'Using {len(self.img_paths)} images from {data_dir}.')

        self.data_len = len(self.img_paths)
        self.num_workers = num_workers
        self.worker_id = int(os.environ.get('RANK', 0))
        self.start, self.restart, self.end = self.distribute()

        # Distribute the dataset accross workers
        self.batch_iterator = iter(self.yield_batches(self.restart, self.end))


    @abstractmethod
    def yield_batches(self, start: int, end: int) -> Generator[TorchTrainingBatch, None, None]:
        pass

    def __iter__(self):
        return self.batch_iterator

    def reset(self):
        self.batch_iterator = iter(self.yield_batches(self.start, self.end))


    def distribute(self) -> Tuple[int, int, int]:
        """Distributes the datasets accross workers.

        Note:
            See https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for more information.

        Returns:
            The default start, re-start (which corresponds to the defaults + the number of steps already run) and end indices for the current worker.
        """
        logger.info(f'Distributing data across {self.num_workers} workers')

        # Compute the number of samples per worker, leaving the last worker with the remainder
        samples_per_worker = self.data_len // self.num_workers
        if self.worker_id == self.num_workers - 1:
            samples_per_worker += self.data_len % self.num_workers
        # Compute the start, re-start and end indices for the current worker
        worker_default_start = self.worker_id * samples_per_worker
        worker_restart = worker_default_start + self.per_worker_steps_run
        worker_end = min(worker_default_start + samples_per_worker, self.data_len)

        logger.info(f'Worker {self.worker_id} is starting at step {worker_restart}')

        return worker_default_start, worker_restart, worker_end


class TorchTrainingDataset(TorchOcrDataset):
    """Dataset for OCR training.

    Warning:
        ``TorchTrainingDataset`` is an infinite iterable dataset, and as such, it does not have a ``__len__`` method. This means that:

            * ``TorchTrainingDataset`` it cannot be used with a ``torch.utils.data.DataLoader`` with ``batch_size > 1``. Actually, \
            ``TorchTrainingDataset.__iter__()`` already returns batches of size inferior to ``max_batch_size`` at each iteration.
            * ``TorchTrainingDataset`` is infinite: Use it with ``next(iter())`` in a ``for`` loop, or with a defined range.
            * ``TorchTrainingDataset.date_len`` is **not** the length of the dataset, but the length of the list of images the dataset will infinitely iterate on. It is therefore not the same as ``__len__``.


    Args:
        data_dir: The directory containing the images to train on.
        max_batch_size: The maximum ocr_batch size to return at each iteration.
        img_height: The height to which to resize the images.
        chunk_width: The width of the chunks to extract from the images.
        chunk_overlap: The overlap between the chunks.
        shuffle: Whether to shuffle the dataset.
        per_worker_steps_run: The number of steps already run by each worker. This is used to compute the number of chunks to
            skip at the beginning of the dataset, so that each worker starts at a different point in the dataset.

    """

    def __init__(self,
                 classes_to_indices: Dict[str, int],
                 max_batch_size: int,
                 img_height: int,
                 chunk_width: int,
                 chunk_overlap: int,
                 special_mapping: Dict[str, str] = None,
                 data_dir: Path = None,
                 img_paths: Optional[List[Path]] = None,
                 loop_infinitely: bool = True,
                 shuffle: bool = True,
                 num_workers: int = 1,
                 per_worker_steps_run: int = 0):

        super().__init__(max_batch_size=max_batch_size,
                         img_height=img_height,
                         chunk_width=chunk_width,
                         chunk_overlap=chunk_overlap,
                         data_dir=data_dir,
                         img_paths=img_paths,
                         num_workers=num_workers,
                         per_worker_steps_run=per_worker_steps_run)

        self.classes_to_indices = classes_to_indices
        self.special_mapping = special_mapping
        self.loop_infinitely = loop_infinitely

        if shuffle:
            random.shuffle(self.img_paths)


    def yield_batches(self, start: int, end: int) -> Generator[TorchTrainingBatch, None, None]:
        """Yields batches of data, starting at the given start index.

        Args:
            start: The start index.
            end: The end index (i.e. the index of the last file to be processed).
        """
        batch_size = 0
        batch_lines = []

        for img_path in self.img_paths[start:end]:
            ocr_line = TorchTrainingLine(img_path, self.img_height, self.chunk_width, self.chunk_overlap, self.classes_to_indices,
                                         special_mapping=self.special_mapping)

            if batch_size + ocr_line.chunks.shape[0] > self.max_batch_size:
                yield TorchTrainingBatch.from_lines(batch_lines)
                batch_lines = [ocr_line]
                batch_size = ocr_line.chunks.shape[0]
            else:
                batch_lines.append(ocr_line)
                batch_size += ocr_line.chunks.shape[0]

        if self.loop_infinitely:
            self.reset()

        yield TorchTrainingBatch.from_lines(batch_lines)


class TorchInferenceDataset(TorchOcrDataset):

    def __init__(self,
                 max_batch_size: int,
                 img_height: int,
                 chunk_width: int,
                 chunk_overlap: int,
                 line_detector: Callable,
                 img_paths: Optional[List[Path]] = None,
                 data_dir: Optional[Path] = None,
                 num_workers: int = 1):

        super().__init__(max_batch_size=max_batch_size,
                         img_height=img_height,
                         chunk_width=chunk_width,
                         chunk_overlap=chunk_overlap,
                         data_dir=data_dir,
                         img_paths=img_paths,
                         num_workers=num_workers)

        self.line_detector = line_detector

    def yield_batches(self, start: int, end: int) -> Generator[TorchInferenceBatch, None, None]:
        """Yields batches of data, starting at the given start index.

        Args:
            start: The start index.
            end: The end index (i.e. the index of the last file to be processed).
        """

        batch_size = 0
        batch_lines = []

        for img_path in self.img_paths[start:end]:

            img_tensor = read_image(str(img_path), mode=ImageReadMode.GRAY).requires_grad_(False)
            line_boxes: List['Shape'] = self.line_detector(img_path)

            page_lines = {f'{img_path.stem}_{"_".join(l.xyxy)}': img_tensor[l.ymin:l.ymax, l.xmin:l.xmax].clone()
                          for l in line_boxes}

            page_lines = [TorchInferenceLine(line_id=k, img_tensor=v, img_height=self.img_height, chunk_width=self.chunk_width)
                          for k, v in page_lines.items()]

            for ocr_line in page_lines:

                if batch_size + ocr_line.chunks.shape[0] > self.max_batch_size:
                    yield TorchInferenceBatch.from_lines(batch_lines)
                    batch_lines = [ocr_line]
                    batch_size = ocr_line.chunks.shape[0]
                else:
                    batch_lines.append(ocr_line)
                    batch_size += ocr_line.chunks.shape[0]

        if batch_lines:
            yield TorchInferenceBatch.from_lines(batch_lines)


class TorchBatchedTrainingDataset(torch.utils.data.Dataset):
    def __init__(self,
                 source_dir: Path,
                 cache_dir: Optional[Path] = None,
                 num_workers: int = 1,
                 epoch_steps_run_per_worker: int = 0,
                 chars_to_special_classes: Dict[str, str] = None,  # Todo this can be removed after first run
                 classes_to_indices: Dict[str, int] = None,  # Todo this can be removed after first run
                 drop_remainding_batches: bool = False):
        """A dataset that loads batches of ocr data from disk.

        Args:
            source_dir: The directory containing the batches.
            cache_dir: The directory to cache the batches in.
            num_workers: The number of workers to use.
            epoch_steps_run_per_worker: The number of steps already run in the current epoch, per worker.
            chars_to_special_classes: A mapping from characters to special classes.
            classes_to_indices: A mapping from classes to indices.
        """
        super().__init__()
        self.source_dir = source_dir
        self.cache_dir = cache_dir
        self.num_workers = num_workers
        self.epoch_steps_run_per_worker = epoch_steps_run_per_worker
        self.chars_to_special_classes = chars_to_special_classes
        self.classes_to_indices = classes_to_indices
        self.drop_remainding_batches = drop_remainding_batches

        self.batch_paths = sorted(source_dir.glob('*.pt'), key=lambda x: int(x.stem))
        self.rank = int(os.environ.get('RANK', 0))

        if self.num_workers > 1:
            self.distribute()

        self.uses_cached_batches = False

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.fetch_function = self.fetch_and_cache_batch
        else:
            self.fetch_function = self.fetch_batch


    #@profile
    def __getitem__(self, idx):
        return self.fetch_function(idx)

    def __len__(self):
        return len(self.batch_paths)

    def fetch_batch(self, idx):
        return TorchTrainingBatch(**torch.load(self.batch_paths[idx]))


    def fetch_and_cache_batch(self, idx):
        batch = self.fetch_batch(idx)
        batch.change_texts(chars_to_special_classes=self.chars_to_special_classes, classes_to_indices=self.classes_to_indices)  # Todo: this is a hack
        torch.save(batch.to_dict(), self.cache_dir / self.batch_paths[idx].name)
        return batch


    def use_cached_batches(self):
        """Uses the cached batches if they are available."""
        if self.cache_dir is not None:
            logger.info(f'Using cached batches from {self.cache_dir}')
            cached_batch_names = [p.name for p in self.cache_dir.glob('*.pt')]

            # Caching missing files
            for i in tqdm(range(len(self.batch_paths)), desc='Caching missing batches'):
                if self.batch_paths[i].name not in cached_batch_names:
                    self.fetch_and_cache_batch(i)
            self.batch_paths = [self.cache_dir / p.name for p in self.batch_paths]
            self.fetch_function = self.fetch_batch
            self.uses_cached_batches = True


    def reset(self):
        """Resets the dataset, so that the next call to __getitem__ will return the first ocr_batch again."""
        logger.info('Resetting dataset')
        self.batch_paths = sorted(self.source_dir.glob('*.pt'), key=lambda x: int(x.stem))
        self.epoch_steps_run_per_worker = 0

        if self.num_workers > 1:
            self.distribute()

        if self.uses_cached_batches:
            self.use_cached_batches()

    def distribute(self):
        """Distributes the datasets accross workers.

        Note:
            See https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for more information.

        Returns:
            The default start, re-start (which corresponds to the defaults + the number of steps already run) and end indices for the current worker.
        """
        logger.info(f'Distributing data across {self.num_workers} workers')

        # Compute the number of samples per worker, leaving the last worker with the remainder
        samples_per_worker = len(self.batch_paths) // self.num_workers
        start = self.rank * samples_per_worker

        if self.rank == self.num_workers - 1 and not self.drop_remainding_batches:
            samples_per_worker += len(self.batch_paths) % self.num_workers

        end = start + samples_per_worker
        start += self.epoch_steps_run_per_worker

        self.batch_paths = self.batch_paths[start:end]


def prepare_text(text: str, special_mapping: dict) -> str:
    """Prepares a text for training.

    Args:
        text: The text to prepare.
        special_mapping: A mapping of special characters to replace, e.g. {'ª': '\x02a'}.
    """
    text = unicodedata.normalize('NFD', harmonise_unicode(text))
    for k, v in special_mapping.items():
        text = text.replace(k, v)
    return text


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


def chunk_img_tensor(img_tensor: torch.Tensor,
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



# #@profile
def prepare_img_tensor(img_tensor: torch.Tensor,
                       img_height: int) -> torch.Tensor:
    """Prepares an image tensor for training.

    Args:
        img_tensor: The tensor of a grayscale image tensor of shape (1, initial_height, initial_width).
        img_height: The height to which to resize the image.

    Returns:
        The prepared image tensor, in shape (1, img_height, resized_width).
    """
    img_tensor = invert_image_tensor(img_tensor)
    # img_tensor = crop_image_tensor_to_nonzero(img_tensor)

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

    # 👁️ the end of this tensor could be chunked to non-zero values to recut the introduced padding
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


def get_custom_dataloader(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=None,
                                       batch_sampler=None,
                                       num_workers=0,
                                       # prefetch_factor=2,
                                       collate_fn=lambda x: x, )


def get_weighted_filelists(filelists_dir: Path,
                           datasets_weights: Dict[str, int],
                           root_dir: Path,
                           splits: Tuple[str] = ('train', 'val', 'test')) -> Dict[str, List[Path]]:
    """Gets the filelists from the given directory, with the given weights.

    Note:
        Each filelist is a .txt file containing the RELATIVE paths to the images in the dataset.

    Args:
        filelists_dir: The directory containing the filelists as .txt files.
        datasets_weights: The weights to apply to each dataset.
        root_dir: The root directory containing the datasets.

    Returns:
        A dictionary containing the absolute paths as `pathlib.Path` for each split.
    """

    logger.info(f'Getting filelists from {filelists_dir} with weights {datasets_weights}')
    imgs_paths = {split: [] for split in splits}

    for txt_path in sorted(filelists_dir.glob('*.txt'), key=lambda x: x.stem):
        split = [s for s in splits if txt_path.stem.endswith(s)][0]
        dataset_name = txt_path.stem.replace(f'_{split}', '')
        dataset_imgs_paths = [root_dir / p for p in txt_path.read_text(encoding='utf-8').splitlines()]

        if split == 'train':
            dataset_imgs_paths *= datasets_weights.get(dataset_name, 1)
        imgs_paths[split] += dataset_imgs_paths

    return imgs_paths


def pre_batch_filelist(filelist: List[Path],
                       config: Dict[str, Any],
                       output_dir: Path,
                       shuffle: bool = True,
                       restart: bool = True):
    """A fault-resistent function to pre-batch a dataset to ``.pt`` files given a config file."""

    if shuffle:
        random.seed(config['random_seed'])
        random.shuffle(filelist)

    if restart:
        last_file_index = max([int(p.stem) for p in output_dir.glob('*.pt')]) + 1
        filelist = filelist[last_file_index:]
        logger.info(f'Restarting from {last_file_index}')
    else:
        last_file_index = 0

    # Start the main for loop to create the batches
    # Todo: this should be done using an ``TorchTrainingDataset``
    batch_size = 0
    ocr_lines = []

    for i, img_path in tqdm(enumerate(filelist, start=last_file_index)):
        if not img_path.exists():
            continue
        if not img_path.with_suffix('.txt').exists():
            continue
        ocr_line = TorchTrainingLine(img_path, config['chunk_height'], config['chunk_width'], config['chunk_overlap'], config['classes_to_indices'],
                                     special_mapping=config['chars_to_special_classes'])

        if batch_size + ocr_line.chunks.shape[0] > config['max_batch_size']:
            torch.save(TorchTrainingBatch.from_lines(ocr_lines).to_dict(), output_dir / f'{i}.pt')
            ocr_lines = [ocr_line]
            batch_size = ocr_line.chunks.shape[0]
        else:
            ocr_lines.append(ocr_line)
            batch_size += ocr_line.chunks.shape[0]

    if ocr_lines:
        torch.save(TorchTrainingBatch.from_lines(ocr_lines).to_dict(), output_dir / f'{i}.pt')


def pre_batch_dataset(config: dict,
                      splits: List[str],
                      output_dir: Path,
                      shuffle: bool = True,
                      restart: bool = True):
    """A fault-resistent function to pre-batch a dataset to ``.pt`` files given a config file."""

    # Get the filelists
    split_filelists = get_weighted_filelists(filelists_dir=config['filelists_dir'],
                                             datasets_weights=config['datasets_weights'],
                                             root_dir=config['datasets_root_dir'], )

    if shuffle:
        random.seed(config['random_seed'])
        for split in splits:
            random.shuffle(split_filelists[split])

    for split in splits:
        logger.info(f'Pre-batching {split} dataset')

        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        img_paths = split_filelists[split]

        if restart:
            last_file_index = max([int(p.stem) for p in split_output_dir.glob('*.pt')]) + 1
            img_paths = img_paths[last_file_index:]
            logger.info(f'Restarting from {last_file_index}')
        else:
            last_file_index = 0

        # Start the main for loop to create the batches
        # Todo: this should be done using an ``TorchTrainingDataset``
        batch_size = 0
        ocr_lines = []

        for i, img_path in tqdm(enumerate(img_paths, start=last_file_index)):
            if not img_path.exists():
                continue
            if not img_path.with_suffix('.txt').exists():
                continue
            ocr_line = TorchTrainingLine(img_path, config['chunk_height'], config['chunk_width'], config['chunk_overlap'],
                                         config['classes_to_indices'],
                                         special_mapping=config['chars_to_special_classes'])

            if batch_size + ocr_line.chunks.shape[0] > config['max_batch_size']:
                torch.save(TorchTrainingBatch.from_lines(ocr_lines).to_dict(), split_output_dir / f'{i}.pt')
                ocr_lines = [ocr_line]
                batch_size = ocr_line.chunks.shape[0]
            else:
                ocr_lines.append(ocr_line)
                batch_size += ocr_line.chunks.shape[0]

        if ocr_lines:
            torch.save(TorchTrainingBatch.from_lines(ocr_lines).to_dict(), split_output_dir / f'{i}.pt')


def batch_builder(ocr_lines: Iterable['TorchLine'],
                  max_batch_size: int,
                  batch_class=TorchTrainingBatch):
    """A generator that yields batches of lines from the given iterable of lines.

    Args:
        ocr_lines: The iterable of ocr lines to batch.
        max_batch_size: The maximum size of the batch.
        batch_class: The class to use to create the batches (should be a ``TorchTrainingBatch`` or a ``TorchInferenceBatch``).
    """

    batch_size = 0
    batch_lines = []

    for ocr_line in ocr_lines:
        if batch_size + ocr_line.chunks.shape[0] > max_batch_size:
            yield batch_class.from_lines(batch_lines)
            batch_lines = [ocr_line]
            batch_size = ocr_line.chunks.shape[0]
        else:
            batch_lines.append(ocr_line)
            batch_size += ocr_line.chunks.shape[0]

    if batch_lines:
        yield batch_class.from_lines(batch_lines)
