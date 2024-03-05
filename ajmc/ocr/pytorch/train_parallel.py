import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Dict, Union

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
from ajmc.ocr.pytorch.config import get_config, write_config_to_output_dir
from ajmc.ocr.pytorch.data_processing import TorchTrainingBatch, recompose_batched_chunks, get_custom_dataloader, TorchBatchedTrainingDataset
from ajmc.ocr.pytorch.model import OcrTorchModel


class OcrModelTrainer:
    def __init__(self,
                 model: OcrTorchModel,
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 validation_rate: int,
                 save_rate: int,
                 output_dir: Path,
                 chunk_overlap: int,
                 device: Optional[Union[torch.device, int]] = None,
                 num_workers: int = 1,
                 epochs: int = 1,
                 total_steps_run: int = 0,
                 epoch_steps_run: int = 0,
                 epochs_run: int = 0):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.validation_rate = validation_rate
        self.save_rate = save_rate
        self.output_dir = output_dir
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.num_workers = num_workers

        self.epochs = epochs
        self.total_steps_run = total_steps_run
        self.epoch_steps_run = epoch_steps_run
        self.epochs_run = epochs_run
        self.epochs_to_run = epochs - epochs_run

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        self.is_distributed = self.num_workers > 1
        self.is_main_process = not self.is_distributed or self.device == 0

        self.best_cer = 0.2


    def compute_batch_outputs(self, x: torch.Tensor, chunks_to_img_mapping: List[int]) -> torch.Tensor:
        """Return the logsoftmaxed model outputs."""
        # N_chunks x W_chunks x N_classes
        x = self.model(x.to(self.device))
        # Here we recompose the outputs
        x = recompose_batched_chunks(x, mapping=chunks_to_img_mapping, chunk_overlap=self.chunk_overlap)
        # We apply the logsoftmax
        return F.log_softmax(x, 2)

    def compute_outputs_loss(self, x: torch.Tensor, batch: TorchTrainingBatch) -> torch.Tensor:
        # Here we need to transpose the outputs to be of shape T x N x C
        return self.criterion(x.permute(1, 0, 2), batch.texts_tensor.to(self.device), batch.img_widths, batch.text_lengths)

    def compute_batch_loss(self, batch: TorchTrainingBatch) -> torch.Tensor:
        """Forward a ocr_batch through the model."""
        return self.compute_outputs_loss(self.compute_batch_outputs(batch.chunks, batch.chunks_to_img_mapping), batch)

    def run_batch(self, batch: TorchTrainingBatch) -> float:
        self.optimizer.zero_grad()
        loss = self.compute_batch_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        return loss.item()

    def progress_bar(self, iterable, desc: str = None, total: int = None):
        return tqdm(iterable, desc=desc, total=total) if self.is_main_process else iterable


    #@profile
    def evaluate_during_training(self) -> Optional[Dict[str, float]]:

        groundtruth_lines: List[str] = []
        predicted_lines: List[str] = []
        worker_running_val_loss = 0.0

        # ========================= Evaluation on validation set =========================
        self.model.eval()
        for batch in self.progress_bar(self.val_dataloader, desc=f'Evaluating model on validation set at step {self.total_steps_run}...'):
            outputs = self.compute_batch_outputs(batch.chunks, batch.chunks_to_img_mapping)
            worker_running_val_loss += self.compute_outputs_loss(outputs, batch).item()

            if self.is_distributed:
                predicted_lines += self.model.module.ctc_decoder.decode(outputs, sizes=batch.img_widths)[0]
            else:
                predicted_lines += self.model.ctc_decoder.decode(outputs, sizes=batch.img_widths)[0]

            groundtruth_lines += batch.texts
        self.model.train()
        # ========================= Compute average validation loss =========================
        worker_avg_val_loss = worker_running_val_loss / len(self.val_dataloader)

        if self.is_distributed:
            allworkers_avg_val_losses = [None for _ in range(self.num_workers)]
            barrier()
            all_gather_object(allworkers_avg_val_losses, worker_avg_val_loss)

        else:
            allworkers_avg_val_losses = [worker_avg_val_loss]

        if self.is_main_process:
            allworkers_avg_val_loss = sum(allworkers_avg_val_losses) / len(allworkers_avg_val_losses)
            wandb.log({'validation_loss': allworkers_avg_val_loss}, step=self.total_steps_run)
            self.scheduler.step(allworkers_avg_val_loss)

        # ========================= Compute and log CER and WER =========================
        if self.is_distributed:
            all_predicted_lines = [None for _ in range(self.num_workers)]
            all_gather_object(all_predicted_lines, predicted_lines)
            all_groundtruth_lines = [None for _ in range(self.num_workers)]
            all_gather_object(all_groundtruth_lines, groundtruth_lines)

        else:
            all_groundtruth_lines = [groundtruth_lines]
            all_predicted_lines = [predicted_lines]

        if self.is_main_process:
            # Log the predictions and the results to wandb
            all_groundtruth_lines = [line for line_list in all_groundtruth_lines for line in line_list]
            all_predicted_lines = [line for line_list in all_predicted_lines for line in line_list]

            preds_table = wandb.Table(columns=['Groundtruth (First 300)', 'Prediction (First 300)'],
                                      data=[[gt_l, pred_l] for gt_l, pred_l in zip(all_groundtruth_lines[:300], all_predicted_lines[:300])])
            wandb.log({f'val_outputs_{self.total_steps_run}': preds_table}, step=self.total_steps_run)
            results = line_based_evaluation(all_groundtruth_lines, all_predicted_lines,
                                            output_dir=(self.output_dir / f'eval_{self.total_steps_run}'))[2]
            results_table = wandb.Table(columns=['Metric', 'Score'], data=[[k, v] for k, v in results.items()])
            wandb.log({f'val_results_{self.total_steps_run}': results_table}, step=self.total_steps_run)
            wandb.log({'chars_ER': ['chars_ER']}, step=self.total_steps_run)

            return results

    def save_model(self, prefix: str = 'last'):
        model_snapshot_path = self.output_dir / f'{prefix}_model.pt'
        torch.save({'MODEL_STATE': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
                    'TOTAL_STEPS_RUN': self.total_steps_run,
                    'EPOCH_STEPS_RUN': self.epoch_steps_run,
                    'EPOCHS_RUN': self.epochs_run,
                    'SCHEDULER_STATE': self.scheduler.state_dict()},
                   model_snapshot_path)
        logger.debug(f'Step {self.total_steps_run} | Training snapshot saved at {model_snapshot_path}')

    #@profile
    def train(self):

        if self.is_main_process:
            logger.info(f'################ Starting training #####################')
            logger.info(f'{self.epochs} epochs in total | {self.epochs_run} already run | {self.epochs_to_run} to go')
            logger.info(f'{self.total_steps_run} steps run in total | {self.epoch_steps_run} run in last epoch')

        worker_running_train_loss = 0.0
        worker_running_steps = 0

        for epoch_num in range(self.epochs_run, self.epochs):
            for batch in self.progress_bar(self.train_dataloader):
                worker_running_train_loss += self.run_batch(batch)
                worker_running_steps += 1
                self.total_steps_run += self.num_workers
                self.epoch_steps_run += self.num_workers

                # ======================= VALIDATION AND LOSS LOGGING =======================
                if self.total_steps_run % self.validation_rate < self.num_workers:
                    # ================ LOG AVERAGE TRAINING LOSS ================
                    worker_avg_train_loss = worker_running_train_loss / worker_running_steps
                    worker_running_train_loss = 0.0
                    worker_running_steps = 0

                    if self.is_distributed:
                        allworkers_avg_train_losses = [None for _ in range(self.num_workers)]
                        barrier()
                        all_gather_object(allworkers_avg_train_losses, worker_avg_train_loss)

                    else:
                        allworkers_avg_train_losses = [worker_avg_train_loss]

                    if self.is_main_process:
                        allworkers_avg_train_loss = sum(allworkers_avg_train_losses) / self.num_workers
                        wandb.log({'training_loss': allworkers_avg_train_loss}, step=self.total_steps_run)

                    # ================ VALIDATION ================
                    results = self.evaluate_during_training()
                    if self.is_main_process:
                        if results['chars_ER'] < self.best_cer:
                            self.best_cer = results['chars_ER']
                            self.save_model(prefix='best')

                    if not self.val_dataloader.dataset.uses_cached_batches:
                        self.val_dataloader.dataset.use_cached_batches()

                if self.is_main_process and self.total_steps_run % self.save_rate < self.num_workers:
                    self.save_model(prefix='last')

            if self.train_dataloader.dataset.epoch_steps_run_per_worker > 0:
                self.train_dataloader.dataset.reset()

            if not self.train_dataloader.dataset.uses_cached_batches:
                self.train_dataloader.dataset.use_cached_batches()

            self.epochs_run += 1
            self.epoch_steps_run = 0


if __name__ == '__main__':

    # ======== Get distributed info ==================
    num_workers = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = num_workers > 1
    rank = int(os.environ.get('RANK', 0))
    is_main_process = not is_distributed or rank == 0

    # ============== Parse the arguments ==================
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the config file', required=True)
    parser.add_argument('--debug', action='store_true', help='Run in debug mode', required=False, default=False)
    parser.add_argument('--no_wandb', action='store_true', help='Do not use wandb', required=False, default=False)
    args = parser.parse_args()

    # ========== Initialize logging ==================
    ROOT_LOGGER.setLevel('DEBUG' if args.debug else 'INFO' if is_main_process else 'ERROR')
    logger = get_ajmc_logger(__name__)

    # =========== Initialize distributed training =================
    if is_distributed:
        logger.info(f'Initializing distributed training with {num_workers} workers...')
        init_process_group(backend='nccl')
        torch.cuda.set_device(rank)

    # =============== Get the config ======================
    config = get_config(Path(args.config_path))
    if is_main_process:
        config['output_dir'].mkdir(exist_ok=True, parents=True)
        write_config_to_output_dir(config)

    # ================= Set random seeds ====================
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed_all(config['random_seed'])

    # ========= Initialize wandb ==================
    if is_main_process:
        wandb.init(project='ocr_' + config['output_dir'].name,
                   name=config['output_dir'].name,
                   config={k: v for k, v in config.items() if 'classes' not in k},
                   mode='disabled' if args.no_wandb else None,
                   resume=True)

    # ============= CREATE OR LOAD MODEL ==================
    total_steps_run = 0
    epoch_steps_run = 0
    epochs_run = 0

    model = OcrTorchModel(config)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_gamma'], patience=config['scheduler_patience'], min_lr=0.00001)

    if config.get('load_from_path', None) is not None and config['load_from_path'].exists():

        logger.info(f'Loading snapshot from {config["load_from_path"]}')

        if is_distributed:
            model_snapshot = torch.load(config['load_from_path'], map_location='cuda:0')
        else:
            model_snapshot = torch.load(config['load_from_path'])

        total_steps_run = model_snapshot['TOTAL_STEPS_RUN']
        epoch_steps_run = model_snapshot['EPOCH_STEPS_RUN'] + config['save_rate'] + num_workers  # Adding this to skip buggy batches
        epochs_run = model_snapshot['EPOCHS_RUN']

        model.load_state_dict(model_snapshot['MODEL_STATE'])
        scheduler.load_state_dict(model_snapshot['SCHEDULER_STATE'])

        del model_snapshot

    # Initialize distributed training
    if is_distributed:
        model = model.to(rank)
        model = DistributedDataParallel(model, device_ids=[rank])
        device = rank

    else:
        device = torch.device(config['device'])
        model = model.to(device)

    # ============= CREATE DATASETS AND DATALOADERS ==================
    train_dataset = TorchBatchedTrainingDataset(source_dir=config['train_data_dir'],
                                                cache_dir=config['cache_dir'] / 'train' if config['cache_dir'] is not None else None,
                                                num_workers=num_workers,
                                                epoch_steps_run_per_worker=epoch_steps_run // num_workers,
                                                chars_to_special_classes=config['chars_to_special_classes'],
                                                classes_to_indices=config['classes_to_indices'],
                                                drop_remainding_batches=True)

    val_dataset = TorchBatchedTrainingDataset(source_dir=config['val_data_dir'],
                                              cache_dir=config['cache_dir'] / 'val' if config['cache_dir'] is not None else None,
                                              num_workers=num_workers,
                                              chars_to_special_classes=config['chars_to_special_classes'],
                                              classes_to_indices=config['classes_to_indices'])

    train_dataloader = get_custom_dataloader(train_dataset)
    val_dataloader = get_custom_dataloader(val_dataset)

    # ====== Initialize the trainer and start training ==============
    trainer = OcrModelTrainer(model=model,
                              train_dataloader=train_dataloader,
                              val_dataloader=val_dataloader,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              output_dir=config['output_dir'],
                              validation_rate=config['validation_rate'],
                              save_rate=config['save_rate'],
                              chunk_overlap=config['chunk_overlap'],
                              device=device,
                              num_workers=num_workers,
                              epochs=config['epochs'],
                              epoch_steps_run=epoch_steps_run,
                              total_steps_run=total_steps_run,
                              epochs_run=epochs_run)

    trainer.train()

    # Clean up distributed training
    if is_distributed:
        destroy_process_group()
