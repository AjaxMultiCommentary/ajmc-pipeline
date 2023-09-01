from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.model import DenseAttention

# Get the config: will be done automatically in the future

config = get_config(Path('ajmc/ocr/pytorch/config.json'))

# Get the model
model = DenseAttention(config)


# Create the Dataset
class OcrTrainingDataset(Dataset):
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
        image = read_image(str(self.img_paths[idx]))  # Todo assert image is in desired format
        text = self.txt_paths[idx].read_text(encoding='utf-8')  # Todo, assert text is in desired format
        labels = torch.tensor([self.classes_to_indices[x] for x in text])

        return image, labels


# Set seed
torch.manual_seed(config['seed'])

# Device
device = torch.device(config['device'])

# Datasets and dataloaders
train_dataset = OcrTrainingDataset(Path(config['train_data_dir']), config['classes'])
test_dataset = OcrTrainingDataset(Path(config['test_data_dir']), config['classes'])

train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

# Model, loss function, optimizer and scheduler
net = DenseAttention(config)
criterion = nn.CTCLoss()
optimizer = optim.SGD(net.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_gamma'], patience=5, min_lr=0.00001)
net.to(device)

BEST_ACCURACY = 0

for epoch in range(config['num_epochs']):  # loop over the dataset multiple times

    running_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        batch_inputs, batch_labels = batch[0].to(device), batch[1].to(device)

        batch_input_lengths = torch.full(size=(config['batch_size'],), fill_value=batch_inputs.shape[1], dtype=torch.long)
        target_lengths = torch.full(size=(config['batch_size'],), fill_value=batch_inputs.shape[1],
                                    dtype=torch.long)  # todo : this will have to be changed
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(batch_inputs)
        loss = criterion(outputs, batch_labels, batch_input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    scheduler.step(running_loss / step)

    print(f'Epoch {epoch + 1} loss: {running_loss / step}')

    # if epoch % 10 == 0:
    #     predictions = torch.tensor([])
    #     labels = torch.tensor([])
    #     with torch.no_grad():
    #         for data in test_dataloader:
    #             batch_inputs, batch_labels = data[0].to(device), data[1]
    #
    #             outputs = net(batch_inputs)
    #             _, batch_predictions = torch.max(outputs.data, 1)  # Prediction is a tensor of size 1
    #
    #             predictions = torch.cat((predictions, batch_predictions.detach().cpu()))
    #             labels = torch.cat((labels, batch_labels))
    #
    #     total_correct = (predictions == labels).sum().item()
    #     total = labels.size(0)
    #     accuracy = total_correct / total
    #     if accuracy > BEST_ACCURACY:
    #         BEST_ACCURACY = accuracy
    #     print('ACCURACY: ', accuracy)

print('Finished Training. Best accuracy: ', BEST_ACCURACY)
