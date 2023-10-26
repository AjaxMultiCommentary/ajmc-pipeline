import argparse
import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import wandb
from torch import nn
from torch.distributed import init_process_group, destroy_process_group, barrier, all_gather_object
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ajmc.commons.miscellaneous import get_ajmc_logger, ROOT_LOGGER
from ajmc.ocr.evaluation import line_based_evaluation
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.data_processing import OcrBatch, recompose_batched_chunks, get_custom_dataloader, BatchedDataset
from ajmc.ocr.pytorch.model import OcrTorchModel

ROOT_LOGGER.setLevel('INFO')
logger = get_ajmc_logger(__name__)


class OcrModelTrainer:
    def __init__(self,
                 model: OcrTorchModel,
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 per_worker_steps_run: int,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 validation_rate: int,
                 save_rate: int,
                 output_dir: Path,
                 chunk_overlap: int,
                 device: Optional[str] = None,
                 num_workers: int = 0,
                 total_steps_run: int = 0):

        self.model = model  # We have to declare a temporary model here, because we need to load the snapshot before we can define the model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.per_worker_steps_run = per_worker_steps_run
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.validation_rate = validation_rate
        self.save_rate = save_rate
        self.output_dir = output_dir
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.num_workers = num_workers
        self.total_steps_run = total_steps_run

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        # Initialize distributed training
        self.is_distributed = self.num_workers > 1
        if self.is_distributed:
            self.device = int(os.environ['LOCAL_RANK'])
            self.model = model.to(self.device)
            self.model = DistributedDataParallel(self.model, device_ids=[self.device])

        else:
            self.device = torch.device(device)
            self.model = model.to(self.device)

        # Updates datasets and create dataloaders
        self.best_cer = 0.3


    def compute_batch_outputs(self, batch: OcrBatch) -> torch.Tensor:
        """Return the logsoftmax of the model outputs."""
        # N_chunks x W_chunks x N_classes
        outputs = self.model(batch.chunks.to(self.device))
        # Here we recompose the outputs
        outputs = recompose_batched_chunks(outputs, mapping=batch.chunks_to_img_mapping, chunk_overlap=self.chunk_overlap)
        # We apply the logsoftmax
        return F.log_softmax(outputs, 2)

    def compute_outputs_loss(self, outputs: torch.Tensor, batch: OcrBatch) -> torch.Tensor:
        # Here we need to transpose the outputs to be of shape T x N x C
        return self.criterion(outputs.permute(1, 0, 2), batch.texts_tensor.to(self.device), batch.img_widths, batch.text_lengths)

    def compute_batch_loss(self, batch: OcrBatch) -> torch.Tensor:
        """Forward a batch through the model."""
        return self.compute_outputs_loss(self.compute_batch_outputs(batch), batch)

    def run_batch(self, batch: OcrBatch) -> float:
        self.optimizer.zero_grad()
        loss = self.compute_batch_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        return loss.item()

    def progress_bar_wrapper(self, iterable, desc: str = None, total: int = None):
        if self.device == 0 or not self.is_distributed:
            return tqdm(iterable, desc=desc, total=total)
        else:
            return iterable


    #@profile
    def evaluate_during_training(self, total_steps: int, epoch_steps: int):

        groundtruth_lines: List[str] = []
        predicted_lines: List[str] = []
        worker_running_val_loss = 0.0

        with torch.no_grad():
            for batch in self.progress_bar_wrapper(self.val_dataloader, desc=f'Evaluating model on validation set at step {total_steps}...'):
                outputs = self.compute_batch_outputs(batch)
                worker_running_val_loss += self.compute_outputs_loss(outputs, batch).item()

                if self.is_distributed:
                    predicted_lines += self.model.module.ctc_decoder.decode(outputs, sizes=batch.img_widths)[0]
                else:
                    predicted_lines += self.model.ctc_decoder.decode(outputs, sizes=batch.img_widths)[0]

                groundtruth_lines += batch.texts

        # Compute average evaluation loss
        worker_avg_val_loss = worker_running_val_loss / len(self.val_dataloader)

        if self.is_distributed:
            barrier()
            # Gather all the average validation losses from all workers
            allworkers_avg_val_losses = [None for _ in range(self.num_workers)]
            all_gather_object(allworkers_avg_val_losses, worker_avg_val_loss)
            # Gather all the predicted and groundtruth lines from all workers
            all_predicted_lines = [None for _ in range(self.num_workers)]
            all_gather_object(all_predicted_lines, predicted_lines)
            all_groundtruth_lines = [None for _ in range(self.num_workers)]
            all_gather_object(all_groundtruth_lines, groundtruth_lines)


        else:
            allworkers_avg_val_losses = [worker_avg_val_loss]
            all_groundtruth_lines = groundtruth_lines
            all_predicted_lines = predicted_lines

        if self.device == 0 or not self.is_distributed:
            # Get the mean average validation loss and log it, run the scheduler
            allworkers_avg_val_loss = sum(allworkers_avg_val_losses) / len(allworkers_avg_val_losses)
            wandb.log({'validation_loss': allworkers_avg_val_loss}, step=total_steps)
            self.scheduler.step(allworkers_avg_val_loss)

            # Log the predictions and the results to wandb
            all_groundtruth_lines = [line for line_list in all_groundtruth_lines for line in line_list]
            all_predicted_lines = [line for line_list in all_predicted_lines for line in line_list]

            preds_table = wandb.Table(columns=['Groundtruth (First 300)', 'Prediction (First 300)'],
                                      data=[[gt_l, pred_l] for gt_l, pred_l in zip(all_groundtruth_lines[:300], all_predicted_lines[:300])])
            wandb.log({f'val_outputs_{total_steps}': preds_table}, step=total_steps)
            results = line_based_evaluation(all_groundtruth_lines, all_predicted_lines, output_dir=(self.output_dir / f'eval_{total_steps}'))[2]
            results_table = wandb.Table(columns=['Metric', 'Score'], data=[[k, v] for k, v in results.items()])
            wandb.log({f'val_results_{total_steps}': results_table}, step=total_steps)
            wandb.log({'chars_ER': ['chars_ER']}, step=total_steps)

            # Save the model if it's the best so far
            if results['chars_ER'] < self.best_cer:
                self.best_cer = results['chars_ER']
                self.save_model(total_steps=total_steps, epoch_steps=epoch_steps, prefix='best')

    def save_model(self, total_steps: int, epoch_steps: int, prefix: str = 'last'):
        model_snapshot = {'MODEL_STATE': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
                          'TOTAL_STEPS_RUN': total_steps,
                          'EPOCH_STEPS_RUN': epoch_steps}
        model_snapshot_path = self.output_dir / f'{prefix}_model.pt'
        torch.save(model_snapshot, model_snapshot_path)
        logger.info(f'Step {total_steps} | Training snapshot saved at {model_snapshot_path}')

    #@profile
    def train(self, epochs: int):
        logger.info(f'Starting training for {epochs} epochs with {self.num_workers} workers')

        worker_running_train_loss = 0.0
        total_steps_run = self.total_steps_run
        epoch_steps = self.per_worker_steps_run * self.num_workers

        for epoch_num in range(epochs):
            for batch in self.progress_bar_wrapper(self.train_dataloader):
                worker_running_train_loss += self.run_batch(batch)
                total_steps_run += self.num_workers
                epoch_steps += self.num_workers

                if total_steps_run % self.validation_rate < self.num_workers:
                    worker_avg_train_loss = worker_running_train_loss / self.validation_rate
                    worker_running_train_loss = 0.0

                    self.evaluate_during_training(total_steps=total_steps_run, epoch_steps=epoch_steps)

                    if self.is_distributed:
                        allworkers_avg_train_losses = [None for _ in range(self.num_workers)]
                        barrier()
                        all_gather_object(allworkers_avg_train_losses, worker_avg_train_loss)

                    else:
                        allworkers_avg_train_losses = [worker_avg_train_loss]

                    if self.device == 0 or not self.is_distributed:
                        allworkers_avg_train_loss = sum(allworkers_avg_train_losses) / len(allworkers_avg_train_losses)
                        wandb.log({'training_loss': allworkers_avg_train_loss}, step=total_steps_run)

                    if not self.val_dataloader.dataset.uses_cached_batches:
                        self.val_dataloader.dataset.use_cached_batches()

                if total_steps_run % self.save_rate < self.num_workers and (self.device == 0 or not self.is_distributed):
                    self.save_model(total_steps=total_steps_run, epoch_steps=epoch_steps, prefix='last')

            if epoch_num == 0 and self.per_worker_steps_run > 0:
                self.train_dataloader.dataset.reset()

            if not self.train_dataloader.dataset.uses_cached_batches:
                self.train_dataloader.dataset.use_cached_batches()

            epoch_steps = 0


def ddp_setup():
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    logger.info(f'Process group initialized with {os.environ["WORLD_SIZE"]} workers.')


def set_rng_seed(rng_seed: int = 0, deterministic: bool = True):
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    torch.cuda.manual_seed_all(rng_seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True


def main(config: dict):
    logger.info(f'Initializing training...')

    # Set random seeds
    set_rng_seed(config['random_seed'])
    # torch.manual_seed(config['random_seed'])

    # Initialize distributed training
    is_distributed = config['num_workers'] > 1
    if is_distributed:
        ddp_setup()

    # Initialize wandb
    if int(os.environ.get('LOCAL_RANK', 0)) == 0 or not is_distributed:
        wandb.init(project='ocr_' + config['output_dir'].name,
                   name=config['output_dir'].name,
                   config={k: v for k, v in config.items() if 'classes' not in k},
                   mode='disabled' if config['debug'] else None)

    # Get the model and the number of steps run
    model = OcrTorchModel(config)
    total_steps_run = 0
    epoch_steps_run = 0
    if config.get('load_from_path', None) is not None:
        logger.info(f'Loading snapshot from {config["load_from_path"]}')
        if is_distributed:
            model_snapshot = torch.load(config['load_from_path'], map_location='cuda:0')
        else:
            model_snapshot = torch.load(config['load_from_path'])
        model.load_state_dict(model_snapshot['MODEL_STATE'])
        total_steps_run = model_snapshot['TOTAL_STEPS_RUN']
        epoch_steps_run = model_snapshot.get('EPOCH_STEPS_RUN', 0) + config['save_rate'] + config['num_workers']  # Adding this to skip buggy batches
        del model_snapshot

    per_worker_steps_run = epoch_steps_run // config['num_workers']
    logger.info(f'per_worker_steps_run {per_worker_steps_run}')

    train_dataset = BatchedDataset(source_dir=config['train_data_dir'],
                                   cache_dir=config['cache_dir'] / 'train' if config['cache_dir'] is not None else None,
                                   num_workers=config['num_workers'],
                                   per_worker_steps_run=per_worker_steps_run)

    val_dataset = BatchedDataset(source_dir=config['val_data_dir'],
                                 cache_dir=config['cache_dir'] / 'val' if config['cache_dir'] is not None else None,
                                 num_workers=config['num_workers'])

    train_dataloader = get_custom_dataloader(train_dataset)
    val_dataloader = get_custom_dataloader(val_dataset)

    # Get the optimizer and the scheduler
    # optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_gamma'], patience=config['scheduler_patience'], min_lr=0.00001)

    # Initialize the trainer and start training
    trainer = OcrModelTrainer(model=model,
                              train_dataloader=train_dataloader,
                              val_dataloader=val_dataloader,
                              per_worker_steps_run=per_worker_steps_run,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              output_dir=config['output_dir'],
                              validation_rate=config['validation_rate'],
                              save_rate=config['save_rate'],
                              chunk_overlap=config['chunk_overlap'],
                              device=config['device'],
                              num_workers=config['num_workers'],
                              total_steps_run=total_steps_run)

    trainer.train(epochs=config['epochs'])

    # Clean up distributed training
    if is_distributed:
        destroy_process_group()


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the config file', required=False)
    parser.add_argument('--debug', action='store_true', help='Run in debug mode', required=False, default=False)
    args = parser.parse_args()

    # Get the config
    config = get_config(Path(args.config_path))
    config['debug'] = args.debug

    if config['debug']:
        ROOT_LOGGER.setLevel('DEBUG')

    main(config)