"""
Files :
    - 1 img : .pt resized, inverted, normalized, chunked, numbered, overlapping
    - 1 txt : utf-8, NFD

Dataset :
    - takes the

Dataloader / sampler :
    - Receives chunks, fills batch until full, returns it.


"""
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


# TODO : remove the following after OcrIterdataset has shown to be working
class OcrTrainingDataset(Dataset):
    def __init__(self,
                 data_dir: Path,
                 classes: List[str],
                 blank: str = '@',
                 txt_ext: str = '.txt'):
        self.data_dir = data_dir
        self.classes = [blank] + classes
        self.pt_paths = list(self.data_dir.rglob('*.pt'))
        self.txt_ext = txt_ext

    @property
    def classes_to_indices(self):
        return {label: i for i, label in enumerate(self.classes)}

    @property
    def indices_to_classes(self):
        return {i: label for label, i in self.classes_to_indices.items()}

    def __len__(self):
        return len(self.pt_paths)

    def __getitem__(self, idx):
        pt_path = self.pt_paths[idx]
        tensors = torch.load(str(pt_path))
        text = pt_path.with_suffix(self.txt_ext).read_text(encoding='utf-8')

        return tensors, text


class OcrTorchDataset(Dataset):
    def __init__(self,
                 data_dir: Path,
                 classes: List[str],
                 blank: str = '@'):
        self.data_dir = data_dir
        self.classes = [blank] + classes
        self.img_paths = list(self.data_dir.glob('*.png'))
        self.txt_paths = [p.with_suffix('.txt') for p in self.img_paths]

    @property
    def classes_to_indices(self):
        return {label: i for i, label in enumerate(self.classes)}

    @property
    def indices_to_classes(self):
        return {i: label for label, i in self.classes_to_indices.items()}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        pass


class OcrTrainingDataset(OcrTorchDataset):

    def __getitem__(self, idx):
        image = read_image(str(self.img_paths[idx]))  # Todo assert image is in desired format
        text = self.txt_paths[idx].read_text(encoding='utf-8')  # Todo, assert text is in desired format
        labels = torch.tensor([self.classes_to_indices[x] for x in text])

        return image, labels


class OcrEvalDataset(OcrTorchDataset):

    def __getitem__(self, idx):
        image = read_image(str(self.img_paths[idx]))  # Todo assert image is in desired format
        text = self.txt_paths[idx].read_text(encoding='utf-8')  # Todo, assert text is in desired format

        return image, text
