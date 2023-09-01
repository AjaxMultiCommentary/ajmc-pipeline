# Todo perform refactoring name after general refactororing.
# Todo rewrite these tests

import pytest
import torch

from ajmc.ocr.pytorch import data_processing as dp
from tests.test_ocr import sample_objects as so
from tests.test_ocr.sample_objects import get_and_write_sample_dataset


config = so.get_sample_config()
single_img_tensor = so.get_single_img_tensor(config)
img_width = single_img_tensor.shape[2]
img_height = single_img_tensor.shape[1]
chunk_width = config['input_shape'][2]
chunk_overlap = config['chunk_overlap']

n_chunks = dp.compute_n_chunks(single_img_tensor, config['input_shape'][2], config['chunk_overlap'])
padding = dp.compute_padding(single_img_tensor, n_chunks, config['input_shape'][2], config['chunk_overlap'])
single_image_chunks = dp.chunk_img_tensor(single_img_tensor, config['input_shape'][2], config['chunk_overlap'])


def test_chunk_img_tensor_with_overlap():
    assert sum([chunk.shape[2] for chunk in single_image_chunks]) >= single_img_tensor.shape[2]
    assert sum([chunk.shape[2] for chunk in single_image_chunks]) - padding < single_img_tensor.shape[2] + config['input_shape'][2]
    assert n_chunks * config['input_shape'][2] - (n_chunks - 1) * config['chunk_overlap'] - single_img_tensor.shape[2] == padding


# rebuild single_image_chunks
def test_recompose_chunks():
    reassembled = dp.recompose_chunks(single_image_chunks, chunk_overlap)
    assert dp.compute_padding(reassembled, n_chunks, chunk_width, chunk_overlap) == 0
    assert reassembled.shape[-1] == single_img_tensor.shape[-1] + padding




def test_OcrIterDataset_iter():
    test_dataset = get_and_write_sample_dataset(3)
    for i, (batch, text, mapping) in enumerate(test_dataset):
        print(f'---------------- batch {i} ----------------')
        print('batch_shape', batch.shape)
        print('batch_text', text)
        print('batch_mapping', mapping)
        assert batch.shape[0] <= test_dataset.max_batch_size
        assert mapping.sum() == batch.shape[0]


@pytest.mark.skip(reason='Not implemented yet, just a manual tester')
def test_OcrDataset_with_dataloader():
    test_dataset = get_and_write_sample_dataset(3)

    dataloader_single = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=0)
    dataloader_distr = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=3)

    for batch in dataloader_distr:
        print(batch[0].shape)
        print(batch[1])
        print("--------------------------------------------")


def test_reassemble_chunks_batch():
    dataset = get_and_write_sample_dataset(3)
    test_batch = next(iter(dataset))
    reassembled = dp.recompose_batched_chunks(test_batch[0], test_batch[-1], chunk_overlap=chunk_overlap)
    assert reassembled.shape[0] == test_batch[-1].shape[0]
    assert reassembled.shape[1] == 1
    assert reassembled.shape[2] == img_height
    assert reassembled.shape[3] == img_width + padding
