from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models.densenet import _Transition, _DenseBlock


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
        self.classifier = nn.Linear(num_features, num_classes)

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
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class CustomImageDataset(Dataset):
    def __init__(self, img_dir: Path, transform=None):
        self.img_dir = img_dir
        self.files = list(self.img_dir.glob('*.png'))
        self.img_labels = [p.stem.split('_')[-1] for p in self.files]
        self.transform = transform


    @property
    def classes(self):
        return (self.img_dir.parent / 'labels.txt').read_text().splitlines()

    @property
    def classes_to_indices(self):
        return {label: i for i, label in enumerate(self.classes)}

    @property
    def indices_to_classes(self):
        return {i: label for label, i in self.classes_to_indices.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = read_image(str(self.files[idx]))
        if self.transform:
            image = self.transform(image)
        image = image.float() / 255
        label = self.classes_to_indices[self.img_labels[idx]]
        label = torch.tensor(label)

        return image, label


# Parameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.2
MOMENTUM = 0.9
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
TRAIN_DATA_DIR = '/scratch/sven/pre_training_images_test/train'
TEST_DATA_DIR = '/scratch/sven/pre_training_images_test/test'
NET_GROWTH_RATE = 12
NET_BLOCK_CONFIG = [3, 3]
NET_NUM_INIT_FEATURES = 128
SEED = 42
SCHEDULER_GAMMA = 0.25
SCHEDULER_STEP_SIZE = 10

# Set seed
torch.manual_seed(SEED)

# Device
device = torch.device(DEVICE)

# Datasets and dataloaders
train_dataset = CustomImageDataset(Path(TRAIN_DATA_DIR), transform=transforms.Resize((40, 100), antialias=True))
test_dataset = CustomImageDataset(Path(TEST_DATA_DIR), transform=transforms.Resize((40, 100), antialias=True))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, loss function, optimizer and scheduler
net = CustomDenseNet(growth_rate=NET_GROWTH_RATE,
                     block_config=NET_BLOCK_CONFIG,
                     num_init_features=NET_NUM_INIT_FEATURES,
                     num_classes=len(train_dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_GAMMA, patience=5, min_lr=0.00001)
net.to(device)

BEST_ACCURACY = 0

for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        batch_inputs, batch_labels = batch[0].to(device), batch[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    scheduler.step(running_loss / step)

    print(f'Epoch {epoch + 1} loss: {running_loss / step}')

    if epoch % 10 == 0:
        predictions = torch.tensor([])
        labels = torch.tensor([])
        with torch.no_grad():
            for data in test_dataloader:
                batch_inputs, batch_labels = data[0].to(device), data[1]

                outputs = net(batch_inputs)
                _, batch_predictions = torch.max(outputs.data, 1)  # Prediction is a tensor of size 1

                predictions = torch.cat((predictions, batch_predictions.detach().cpu()))
                labels = torch.cat((labels, batch_labels))

        total_correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = total_correct / total
        if accuracy > BEST_ACCURACY:
            BEST_ACCURACY = accuracy
        print('ACCURACY: ', accuracy)

print('Finished Training. Best accuracy: ', BEST_ACCURACY)
