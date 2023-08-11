import math
import re
import time
from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from ajmc.ocr import variables as ocr_vs


class OcrLine:
    """A custom class for OCR lines.

    Note:
        - A line corresponds both to a txt and an img if in train mode, only an image if not.
        - img : the image of the line
            - img_width: The width of the original image
            - img_height: The height of the original image
            - img_padding_length: The length of padding added to the image to make it divisible by `chunks_count`
        - Chunks : the chunks of the image of the line
            - chunks_width: The widths of chunks
            - chunks_height = img height
            - chunks_count = The number of chunks
        - text : the text corresponding to the image.
            - text_length: The length of the text

    """

    # Todo test this for speed
    def __init__(self,
                 path: Path,
                 img_height: int,
                 chunk_overlap: int,
                 chunk_width: int,
                 ):
        start_time = time.time()
        img_tensor = read_image(str(path), mode=ImageReadMode.GRAY)
        self.img_width = img_tensor.shape
        tensor = prepare_img_tensor(img_tensor,
                                    img_height=img_height)
        self.chunks, self.img_padding = chunk_img_tensor_with_overlap(tensor, chunk_width, chunk_overlap)
        self.len_chunks = len(self.chunks)
        self.text = path.with_suffix('.gt.txt').read_text(encoding='utf-8')  # Todo change txt
        self.txt_tensor = None

        print('Img initialized in : {:.2f}s'.format(time.time() - start_time))


class OcrBatch:

    def __init__(self):
        self.ocr_lines: List[OcrLine] = []

        self.chunks = torch.cat([img.chunks for img in self.ocr_lines], dim=0)
        self.mappings = [l.len_chunks for l in self.ocr_lines]
        self.txt_lengths = [len(l.txt_tensor) for l in self.ocr_lines]
        self.txt_batch_tensor = pad_sequence([img.txt_tensor for img in self.ocr_lines], batch_first=True, )
        self.img_widths = [l.img_width for l in self.ocr_lines]


# Ce qu'on veut in fine comme batch c'est
#

def detect_spaces(img_tensor: torch.Tensor,
                  height_space_ratio: float = 0.3,
                  height_noise_ratio: float = 0.05) -> list[tuple[int, int]]:
    """Detects spaces in a binary, white-on-black line image tensor.

    This function first finds the columns with only zeros (black pixels), using torch.any. If there are more than
    image height x `height_space_ratio` consecutive zeros columns, then they are considered a space.

    Note:
        This function is not robust to noise.

    Args:
        img_tensor: A binary, white-on-black line image tensor, with shape (1, height, width).
        height_space_ratio: The ratio of the image height that a space must occupy to be considered a space.
        height_noise_ratio: The tolerated number of pixel in a space column, expressed as a ratio of the image height.


    Returns:
        A list of tuples (offset, length) representing the detected spaces.

    """
    img_height = img_tensor.shape[1]
    space_threshold = img_height * height_space_ratio

    are_columns_zeros = torch.where(img_tensor.sum(dim=1) < 255 * img_height * height_noise_ratio, False, True)
    uniques, counts = torch.unique_consecutive(are_columns_zeros, return_counts=True)

    previous_offset = 0
    spaces = []
    for unique, count in zip(uniques.tolist(), counts.tolist()):
        if not unique and count > space_threshold:
            spaces.append((previous_offset, count))
        previous_offset += count

    return spaces


def pad_image_tensor_to_multiple(img_tensor, desired_width: int):
    """Pads an image tensor to a multiple of a desired width."""
    height, width = img_tensor.shape
    if width % desired_width == 0:
        return img_tensor
    else:
        num_cols_to_add = desired_width - (width % desired_width)
        padding = torch.zeros((height, num_cols_to_add))
        return torch.cat((img_tensor, padding), dim=1)


def invert_image_tensor(image_tensor):
    return 255 - image_tensor


# Todo : legacy : delete once `chunk_img_tenso_with_overlap` is tested
def chunk_img_tensor_space_based(image_tensor: torch.Tensor,
                                 chunk_max_width: int,
                                 height_space_proportion: float = 0.3) -> List[torch.Tensor]:
    """Chunks an image tensor into a list of tensors given a desired width."""
    if image_tensor.shape[2] <= chunk_max_width:
        return [image_tensor]

    spaces = detect_spaces(image_tensor, height_space_proportion)
    space_indices = [space[0] + int(space[1] / 2) for space in spaces] + [image_tensor.shape[-1]]

    chunks = []
    chunk_index = 0
    chunk_length = 0
    previous_space_index = 0

    for space_index in space_indices:
        word_length = space_index - previous_space_index

        if word_length > chunk_max_width:
            raise NotImplementedError('A chunk with no spaces is longer than the desired chunk width.')

        chunk_length += word_length
        if chunk_length > chunk_max_width:
            chunks.append(image_tensor[:, :, chunk_index:previous_space_index])
            chunk_index = previous_space_index
            chunk_length = word_length

        previous_space_index = space_index

    if chunk_length:
        chunks.append(image_tensor[:, :, chunk_index:])

    return chunks


# Todo : legacy : delete once `chunk_img_tenso_with_overlap` is tested
def chunk_img_tensor_space_based_with_text(img_tensor: torch.Tensor,
                                           chunk_max_width: int,
                                           text: str,
                                           height_space_ratio: float = 0.3) -> List[Tuple[torch.Tensor, str]]:
    """Chunks a line image tensor and its corresponding text into a list of image and text chunks"""

    # if img_tensor.shape[2] <= chunk_max_width:
    #     return [(img_tensor, text)]

    # Prepare the corresponding text
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    spaces = detect_spaces(img_tensor, height_space_ratio)

    assert len(spaces) == len(words) - 1, """Incorrect alignment between image and text"""

    space_indices = [space[0] + int(space[1] / 2) for space in spaces] + [img_tensor.shape[-1]]

    chunks = []
    chunk_index = 0
    chunk_length = 0
    chunk_words = []
    previous_space_index = 0

    for space_index, word in zip(space_indices, words):
        word_length = space_index - previous_space_index

        if word_length > chunk_max_width:
            raise NotImplementedError('A chunk with no spaces is longer than the desired chunk length.')

        chunk_length += word_length

        if chunk_length > chunk_max_width:
            chunks.append((img_tensor[:, :, chunk_index:previous_space_index], ' '.join(chunk_words)))

            # Reset the counters
            chunk_index = previous_space_index
            chunk_length = word_length
            chunk_words = []

        chunk_words.append(word)
        previous_space_index = space_index

    if chunk_length:
        chunks.append((img_tensor[:, :, chunk_index:], ' '.join(chunk_words)))

    return chunks


def normalize_image_tensor(image_tensor):
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


def chunk_img_tensor_with_overlap(img_tensor,
                                  chunk_width: int,
                                  chunk_overlap: int) -> Tuple[torch.Tensor, int]:
    n_chunks = compute_n_chunks(img_tensor, chunk_width, chunk_overlap)
    padding = compute_padding(img_tensor, n_chunks, chunk_width, chunk_overlap)

    # Pad the tensor
    img_tensor = torch.nn.functional.pad(img_tensor, (0, padding), mode='constant', value=0)

    # Chunk the tensor
    chunks = []
    for i in range(n_chunks):
        chunks.append(img_tensor[:, :, i * (chunk_width - chunk_overlap):i * (chunk_width - chunk_overlap) + chunk_width])

    return torch.stack(chunks, dim=0), padding


def prepare_img_tensor(img_tensor: torch.Tensor,
                       img_height: int) -> torch.Tensor:
    """Prepares an image tensor for training."""

    img_tensor = invert_image_tensor(img_tensor)

    img_tensor = transforms.Resize(img_height)(img_tensor)
    img_tensor = crop_image_tensor_to_nonzero(img_tensor)
    img_tensor = normalize_image_tensor(img_tensor)

    return img_tensor


def prepare_files_for_torch(dataset_dir: Path,
                            output_dir: Path,
                            img_height: int,
                            chunks_width: int,
                            chunks_overlap: int,
                            invert: bool = True,
                            img_extension: str = ocr_vs.IMG_EXTENSION,
                            txt_extension: str = ocr_vs.GT_TEXT_EXTENSION,
                            debug: bool = False):
    """Prepares """

    for img_path in dataset_dir.glob('*' + img_extension):
        img_tensor = read_image(str(img_path), mode=ImageReadMode.GRAY)

        if invert:
            img_tensor = invert_image_tensor(img_tensor)

        img_tensor = transforms.Resize(img_height)(img_tensor)
        img_tensor = crop_image_tensor_to_nonzero(img_tensor)
        img_tensor = normalize_image_tensor(img_tensor)

        chunks_tensor, padding = chunk_img_tensor_with_overlap(img_tensor=img_tensor,
                                                               chunk_width=chunks_width,
                                                               chunk_overlap=chunks_overlap)

        # save
        torch.save(chunks_tensor, (output_dir / (img_path.stem + '.pt')))
        (output_dir / (img_path.stem + txt_extension)).write_text(img_path.with_suffix(txt_extension).read_text(encoding='utf-8'), encoding='utf-8')

        if debug:
            print(img_path)
            transforms.ToPILImage()(img_tensor).show()

            for chunk in chunks_tensor:
                print('chunk')
                transforms.ToPILImage()(chunk).show()
            if input() == 'q':
                break


def reassemble_chunks(chunks: torch.Tensor,
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


def reassemble_chunks_batch(batch: torch.Tensor, mapping: torch.Tensor, chunk_overlap: int) -> torch.Tensor:
    """Apply `reassemble_chunks` to a batch of chunks.

    Args:
        batch: The batch of chunks, in shape (n_chunks, 1, img_height, chunk_width).
        mapping: The mapping from the batch to the original image tensor, in shape (1). The mapping is a tensor of
            integers, where each integer represents the number of chunks that were extracted from the original image
            tensor. Eg : [3, 2] means that the first image tensor was chunked into 3 chunks, the second into 2 chunks.
        chunk_overlap: The overlap between the chunks.

    Returns:
        The reassembled image tensor, in shape (n_images, 1, img_height, img_width+eventual padding).
    """
    chunk_groups = torch.split(batch, mapping.int().tolist(), dim=0)
    reassembled = [reassemble_chunks(group, chunk_overlap) for group in chunk_groups]
    return torch.stack(reassembled, dim=0)


class OcrIterDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 data_dir: Path,
                 classes: str,
                 max_batch_size: int,
                 blank: str = '@',
                 txt_ext: str = '.txt',
                 is_training: bool = True):
        self.data_dir = data_dir
        self.classes = blank + classes
        self.max_batch_size = max_batch_size
        self.pt_paths = sorted(self.data_dir.rglob('*.pt'), key=lambda x: x.stem)
        self.txt_ext = txt_ext
        self.is_training = is_training  # Todo see how we handle this


    @staticmethod
    def get_chunk_size(tensor_path: Path) -> int:
        return int(tensor_path.stem.split('_')[-1])

    @property
    def classes_to_indices(self):
        return {label: i for i, label in enumerate(self.classes)}

    @property
    def indices_to_classes(self):
        return {i: label for label, i in self.classes_to_indices.items()}


    def get_labels_from_pt_path(self, pt_path: Path) -> torch.Tensor:
        text = pt_path.with_suffix(self.txt_ext).read_text(encoding='utf-8')
        return torch.tensor([self.classes_to_indices[x] for x in text]).unsqueeze(0)


    def get_batch_tuple(self,
                        pt_path: Path,
                        batch_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
                                torch.tensor([]), torch.tensor([]), torch.tensor([]))
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        img_batch = torch.cat([batch_tuple[0], torch.load(pt_path)], dim=0)
        labels_batch = torch.stack([batch_tuple[1], self.get_labels_from_pt_path(pt_path)], dim=0)
        labels_to_img_map = torch.cat([batch_tuple[2], torch.tensor([self.get_chunk_size(pt_path)])], dim=0)

        return img_batch, labels_batch, labels_to_img_map


    def distribute(self):
        """Distributes the datasets accross workers, see https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset"""
        start = 0
        end = len(self.pt_paths)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = start + worker_id * per_worker
            end = min(start + per_worker, end)

        return start, end

    def _custom_iterator(self):
        """
        We keep tensors in the following form:
            id_bs0.pt
            id_bs0.txt

        """
        start, end = self.distribute()
        next_index = start
        while next_index < end:

            if self.get_chunk_size(self.pt_paths[next_index]) > self.max_batch_size:  # If there are more chunks that batch_size, skip
                next_index += 1
                continue

            batch_tuple = self.get_batch_tuple(self.pt_paths[next_index])
            next_index += 1
            # we get the number of chunks (=shape[0]) in the next tensor

            while next_index < end and batch_tuple[0].shape[0] + self.get_chunk_size(self.pt_paths[next_index]) <= self.max_batch_size:
                batch_tuple = self.get_batch_tuple(self.pt_paths[next_index], batch_tuple)
                next_index += 1

            yield batch_tuple

    def __iter__(self):
        return self._custom_iterator()
