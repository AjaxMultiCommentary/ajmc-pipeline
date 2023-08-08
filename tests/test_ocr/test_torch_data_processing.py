# Todo perform refactoring name after general refactororing.

import shutil
from pathlib import Path

import pytest
import torch

from ajmc.ocr.torch import data_processing as dp


img_height = 2
img_width = 7
chunk_width = 4
chunk_overlap = 2

single_img_tensor = torch.tensor(list(range(img_width * img_height))).reshape(1, img_height, img_width)
n_chunks = dp.compute_n_chunks(single_img_tensor, chunk_width, chunk_overlap)
padding = dp.compute_padding(single_img_tensor, n_chunks, chunk_width, chunk_overlap)
single_image_chunks = dp.chunk_img_tensor_with_overlap(single_img_tensor, chunk_width, chunk_overlap=chunk_overlap)


def test_chunk_img_tensor_with_overlap():
    assert sum([chunk.shape[2] for chunk in single_image_chunks]) >= single_img_tensor.shape[2]
    assert sum([chunk.shape[2] for chunk in single_image_chunks]) - padding < single_img_tensor.shape[2] + chunk_width
    assert n_chunks * chunk_width - (n_chunks - 1) * chunk_overlap - single_img_tensor.shape[2] == padding


# rebuild single_image_chunks
def test_reassemble_chunks():
    reassembled = dp.reassemble_chunks(single_image_chunks, chunk_overlap)
    assert dp.compute_padding(reassembled, n_chunks, chunk_width, chunk_overlap) == 0
    assert reassembled.shape[-1] == single_img_tensor.shape[-1] + padding


def build_dataset(num_images):
    imgs_batch_tensor = torch.tensor(list(range(num_images * img_width * img_height))).reshape(num_images, 1, img_height, img_width)
    imgs_batch_texts = [f'{i}' for i in range(num_images)]
    imgs_batch_chunks = [dp.chunk_img_tensor_with_overlap(t, chunk_width, chunk_overlap=chunk_overlap) for t in imgs_batch_tensor]

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_']
    max_batch_size = 7

    test_dir = Path('/Users/sven/Desktop/tests/test_ocr_iter_dataset')
    shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(exist_ok=True)

    for i in range(num_images):
        bs = imgs_batch_chunks[i].shape[0]
        torch.save(imgs_batch_chunks[i], test_dir / f'{i}_{bs}.pt')
        (test_dir / f'{i}_{bs}.txt').write_text(imgs_batch_texts[i], encoding='utf-8')

    return dp.OcrIterDataset(test_dir, classes=classes, max_batch_size=max_batch_size)


def test_OcrIterDataset_iter():
    test_dataset = build_dataset(3)
    for i, (batch, text, mapping) in enumerate(test_dataset):
        print(f'---------------- batch {i} ----------------')
        print('batch_shape', batch.shape)
        print('batch_text', text)
        print('batch_mapping', mapping)
        assert batch.shape[0] <= test_dataset.max_batch_size
        assert mapping.sum() == batch.shape[0]


@pytest.mark.skip(reason='Not implemented yet, just a manual tester')
def test_OcrDataset_with_dataloader():
    test_dataset = build_dataset(3)

    dataloader_single = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=0)
    dataloader_distr = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=3)

    for batch in dataloader_distr:
        print(batch[0].shape)
        print(batch[1])
        print("--------------------------------------------")


dataset = build_dataset(3)

test_batch = next(iter(dataset))


def test_reassemble_chunks_batch():
    reassembled = dp.reassemble_chunks_batch(test_batch[0], test_batch[-1], chunk_overlap=chunk_overlap)
    assert reassembled.shape[0] == test_batch[-1].shape[0]
    assert reassembled.shape[1] == 1
    assert reassembled.shape[2] == img_height
    assert reassembled.shape[3] == img_width + padding


test_reassemble_chunks_batch()
