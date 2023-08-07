# Todo, change this file's name after refactoring.
import shutil
from pathlib import Path

import pytest
import torch

from ajmc.ocr.svens_net import datasets as ds


# test OcrIterDataset.custom_iterator


# test OcrIterDataset
@pytest.mark.skip(reason='Just a builder function')
def build_dataset(batch_sizes):
    test_batches = [torch.rand((bs, 2, 4)) if i % 2 == 0 else torch.zeros((bs, 2, 4)) for i, bs in enumerate(batch_sizes)]
    test_texts = [f'{i}_{bs}' for i, bs in enumerate(batch_sizes)]
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '_']
    max_batch_size = 5

    test_dir = Path('/Users/sven/Desktop/test_ocr_iter_dataset')
    shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(exist_ok=True)

    for i, (batch, text, batch_size) in enumerate(zip(test_batches, test_texts, batch_sizes)):
        torch.save(batch, test_dir / f'{i}_{batch_size}.pt')
        (test_dir / f'{i}_{batch_size}.txt').write_text(text, encoding='utf-8')

    return ds.OcrIterDataset(test_dir, classes=classes, max_batch_size=5)


def test_OcrIterDataset_iter():
    test_dataset = build_dataset([6, 3, 4, 1, 2, 3, 2, 2, 4])
    for i, (batch, text, mapping) in enumerate(test_dataset):
        print(f'batch {i}')
        print('batch_shape', batch.shape)
        print('batch_text', text)
        print('batch_mapping', mapping)
        assert batch.shape[0] <= test_dataset.max_batch_size
        assert mapping.sum() == batch.shape[0]


@pytest.mark.skip(reason='Not implemented yet, just a manual tester')
def test_OcrDataset_with_dataloader():
    test_dataset = build_dataset([5, 4, 3, 5, 4, 3, 5, 4, 3])

    dataloader_single = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=0)
    dataloader_distr = torch.utils.data.DataLoader(test_dataset, batch_size=None, num_workers=3)

    for batch in dataloader_distr:
        print(batch[0].shape)
        print(batch[1])
        print("--------------------------------------------")
