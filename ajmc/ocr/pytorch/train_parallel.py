import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.optim as optim
from torch import nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.commons.variables import PACKAGE_DIR
from ajmc.ocr.evaluation import line_based_evaluation
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.data_processing import OcrIterDataset, OcrBatch, recompose_batched_chunks, get_custom_dataloader
from ajmc.ocr.pytorch.model import OcrTorchModel

logger = get_custom_logger(__name__, level=2)


class OcrModelTrainer:
    def __init__(self,
                 model: OcrTorchModel,
                 train_dataset: OcrIterDataset,
                 eval_dataset: OcrIterDataset,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 scheduler_step_rate: int,
                 snapshot_rate: int,
                 snapshot_path: Path,
                 evaluation_rate: int,
                 evaluation_output_dir: Path,
                 chunk_overlap: int,
                 num_workers: int,
                 device: Optional[str] = None):

        self.model = model  # We have to declare a temporary model here, because we need to load the snapshot before we can define the model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_rate = scheduler_step_rate
        self.snapshot_rate = snapshot_rate
        self.snapshot_path = snapshot_path
        self.evaluation_rate = evaluation_rate
        self.evaluation_output_dir = evaluation_output_dir
        self.chunk_overlap = chunk_overlap
        self.num_workers = num_workers
        self.device = device

        self.total_steps_run = 0
        self.per_worker_steps_run = 0

        self.criterion = nn.CTCLoss()
        self.is_distributed = self.num_workers > 0

        # Load snapshot if it exists
        if self.snapshot_path.exists():
            logger.info(f'Loading snapshot from {snapshot_path}')
            if self.is_distributed:
                loc = f"cuda:{self.device}"
                snapshot = torch.load(self.snapshot_path, map_location=loc)
            else:
                snapshot = torch.load(self.snapshot_path)
            self.model.load_state_dict(snapshot["MODEL_STATE"])
            self.total_steps_run = snapshot["TOTAL_STEPS_RUN"]

        # Initialize distributed training
        if self.is_distributed:
            self.device = int(os.environ["LOCAL_RANK"])
            self.model = model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.device])
            self.per_worker_steps_run = self.total_steps_run // self.num_workers

        else:
            self.device = torch.device(device)
            self.model = model.to(self.device)
            self.per_worker_steps_run = self.total_steps_run

        # Updates datasets and create dataloaders
        train_dataset.per_worker_steps_run = self.per_worker_steps_run
        self.train_dataloader = get_custom_dataloader(train_dataset, num_workers=self.num_workers)
        self.eval_dataloader = get_custom_dataloader(eval_dataset, num_workers=self.num_workers)


    def _run_batch(self, batch: OcrBatch) -> float:

        self.optimizer.zero_grad()
        outputs = self.model(batch.chunks.to(self.device))  # N_chunks x W_chunks x N_classes

        # Here we recompose the outputs
        outputs = recompose_batched_chunks(outputs,
                                           mapping=batch.chunks_to_img_mapping,
                                           chunk_overlap=self.chunk_overlap)
        outputs = torch.nn.functional.log_softmax(outputs, dim=2)

        # Here we need to transpose the outputs to be of shape T x N x C
        outputs = outputs.permute(1, 0, 2)

        loss = self.criterion(outputs, batch.texts_tensor.to(self.device), batch.img_widths, batch.text_lengths)
        loss.backward()
        # Todo see if gradient accumulation steps is necessary after testing on RunAI
        self.optimizer.step()
        return loss.item()

    def _convert_targets_to_strings(self, targets):
        strings = []
        for target in targets:
            strings.append(''.join([self.eval_dataloader.dataset.indices_to_classes[char_index.item()] for char_index in target]))

    def _evaluate_during_training(self):

        all_targets = []  # Todo harmonise the name target
        all_predictions: List[str] = []

        with torch.no_grad():
            for step, batch in enumerate(self.eval_dataloader):
                source = batch.chunks.to(self.device)
                all_targets += batch.texts
                if self.is_distributed:
                    all_predictions += self.model.module.predict(source, batch.chunks_to_img_mapping)
                else:
                    all_predictions += self.model.predict(source, batch.chunks_to_img_mapping)

        line_based_evaluation(all_targets, all_predictions, output_dir=self.evaluation_output_dir)


    def _save_snapshot(self, step):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            "TOTAL_STEPS_RUN": step,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Step {step} | Training snapshot saved at {self.snapshot_path}")

    def train(self, total_steps: int):

        per_worker_steps_total = total_steps // self.num_workers if self.is_distributed else total_steps

        running_loss = 0.0

        for step in range(self.per_worker_steps_run, per_worker_steps_total):
            print(f"Worker {self.device} | Step {step} | Running loss: {running_loss} |")
            step *= self.num_workers
            batch = next(iter(self.train_dataloader))
            loss = self._run_batch(batch)
            running_loss += loss

            if step % self.scheduler_step_rate < self.num_workers:
                logger.info(f"Step {step} | Loss: {loss}")
                self.scheduler.step(running_loss / self.scheduler_step_rate)
                running_loss = 0.0

            if step % self.evaluation_rate < self.num_workers and (self.device == 0 or not self.is_distributed):
                self._evaluate_during_training()

            if step % self.snapshot_rate < self.num_workers and (self.device == 0 or not self.is_distributed):
                self._save_snapshot(step)

        self._evaluate_during_training()


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main(config: dict):
    logger.debug(f'Starting training')
    if config['num_workers'] > 0:
        ddp_setup()
        logger.debug(f'Process group initialized')

    # Get the dataset
    train_dataset = OcrIterDataset(data_dir=config['train_data_dir'],
                                   classes=config['classes'],
                                   max_batch_size=config['max_batch_size'],
                                   img_height=config['input_shape'][1],
                                   chunk_width=config['chunk_width'],
                                   chunk_overlap=config['chunk_overlap'],
                                   indices_to_classes=config['indices_to_classes'],
                                   classes_to_indices=config['classes_to_indices'])

    eval_dataset = OcrIterDataset(data_dir=config['test_data_dir'],
                                  classes=config['classes'],
                                  max_batch_size=config['max_batch_size'],
                                  img_height=config['input_shape'][1],
                                  chunk_width=config['chunk_width'],
                                  chunk_overlap=config['chunk_overlap'],
                                  indices_to_classes=config['indices_to_classes'],
                                  classes_to_indices=config['classes_to_indices'])

    model = OcrTorchModel(config)

    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_gamma'], patience=config['scheduler_patience'], min_lr=0.00001)

    trainer = OcrModelTrainer(model=model,
                              train_dataset=train_dataset,
                              eval_dataset=eval_dataset,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              scheduler_step_rate=config['scheduler_step_rate'],
                              snapshot_rate=config['snapshot_rate'],
                              snapshot_path=config['snapshot_path'],
                              evaluation_output_dir=config['evaluation_output_dir'],
                              evaluation_rate=config['evaluation_rate'],
                              chunk_overlap=config['chunk_overlap'],
                              num_workers=config['num_workers'],
                              device=config['device'])

    trainer.train(total_steps=config['total_steps'])
    if config['num_workers'] > 0:
        destroy_process_group()


if __name__ == "__main__":
    # Todo: add argparse to get config path
    config = get_config(PACKAGE_DIR / 'tests/test_ocr/config.json')
    logger.info(f"Running with config: {config}")
    main(config)
