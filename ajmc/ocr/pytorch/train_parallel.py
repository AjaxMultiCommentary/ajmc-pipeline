import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Dict

import torch
import torch.optim as optim
from torch import nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ajmc.commons.miscellaneous import get_ajmc_logger, ROOT_LOGGER
from ajmc.commons.visualisations import draw_lineplot
from ajmc.ocr.evaluation import line_based_evaluation
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.data_processing import OcrIterDataset, OcrBatch, recompose_batched_chunks, get_custom_dataloader
from ajmc.ocr.pytorch.model import OcrTorchModel

ROOT_LOGGER.setLevel('INFO')
logger = get_ajmc_logger(__name__)


class OcrModelTrainer:
    def __init__(self,
                 model: OcrTorchModel,
                 train_dataloader: OcrIterDataset,
                 eval_dataloader: OcrIterDataset,
                 per_worker_steps_run: int,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 scheduler_step_rate: int,
                 snapshot_rate: int,
                 snapshot_path: Path,
                 evaluation_rate: int,
                 evaluation_output_dir: Path,
                 chunk_overlap: int,
                 device: Optional[str] = None,
                 num_workers: int = 0):

        self.model = model  # We have to declare a temporary model here, because we need to load the snapshot before we can define the model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.per_worker_steps_run = per_worker_steps_run
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_rate = scheduler_step_rate
        self.snapshot_rate = snapshot_rate
        self.snapshot_path = snapshot_path
        self.evaluation_rate = evaluation_rate
        self.evaluation_output_dir = evaluation_output_dir
        self.chunk_overlap = chunk_overlap
        self.device = device
        self.num_workers = num_workers

        self.criterion = nn.CTCLoss()
        self.is_distributed = self.num_workers > 1

        # Initialize distributed training
        if self.is_distributed:
            self.device = int(os.environ["LOCAL_RANK"])
            self.model = model.to(self.device)
            self.model = DDP(self.model, device_ids=[self.device])

        else:
            self.device = torch.device(device)
            self.model = model.to(self.device)

        # Updates datasets and create dataloaders
        self.results = None  # Is going to updated later

    def run_batch(self, batch: OcrBatch) -> float:

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

    def convert_targets_to_strings(self, targets):
        strings = []
        for target in targets:
            strings.append(''.join([self.eval_dataloader.dataset.indices_to_classes[char_index.item()] for char_index in target]))

    def evaluate_during_training(self, step):

        logger.info("Evaluating model during training")
        groundtruth_lines = []
        predicted_lines: List[str] = []

        with torch.no_grad():
            for _ in tqdm(range(self.eval_dataloader.dataset.data_len)):
                batch = next(iter(self.eval_dataloader))
                source = batch.chunks.to(self.device)
                groundtruth_lines += batch.texts
                if self.is_distributed:
                    predicted_lines += self.model.module.predict(source, batch.chunks_to_img_mapping)
                    for predicted_line in predicted_lines:
                        logger.info(f'Predicted line: {predicted_line}')
                else:
                    predicted_lines += self.model.predict(source, batch.chunks_to_img_mapping)

        return line_based_evaluation(groundtruth_lines, predicted_lines, output_dir=(self.evaluation_output_dir / f"eval_{step}"))


    def save_snapshot(self, step):
        logger.info(f"Saving snapshot at step {step}")
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            "TOTAL_STEPS_RUN": step,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Step {step} | Training snapshot saved at {self.snapshot_path}")


    def update_results(self, results: Dict[str, float]):

        if self.results is None:
            self.results = {k: [v] for k, v in results.items()}
        else:
            for k, v in results.items():
                self.results[k].append(v)

    def train(self, total_steps: int):

        logger.info(f"Starting training for {total_steps} steps with {self.num_workers} workers")
        per_worker_steps_total = total_steps // self.num_workers

        scheduler_running_loss = 0.0
        eval_running_loss = 0.0
        eval_losses = []
        eval_steps = []

        for step in tqdm(range(self.per_worker_steps_run + 1, per_worker_steps_total + 1)):  # Adding +1 to avoid evaluating at step 0
            step *= self.num_workers

            batch = next(iter(self.train_dataloader))
            loss = self.run_batch(batch)
            # logger.debug(f"Worker {self.device} | Step {step} | Running loss: {loss} |")

            scheduler_running_loss += loss
            eval_running_loss += loss

            if step % self.scheduler_step_rate < self.num_workers:
                logger.info(f"Running Scheduler | Step {step} | Loss: {loss / self.scheduler_step_rate}")
                self.scheduler.step(scheduler_running_loss / self.scheduler_step_rate)
                scheduler_running_loss = 0.0

            if step % self.evaluation_rate < self.num_workers and (self.device == 0 or not self.is_distributed):
                eval_losses.append(eval_running_loss / self.evaluation_rate)
                eval_steps.append(step)
                eval_running_loss = 0.0

                _, _, results = self.evaluate_during_training(step=step)
                self.update_results(results)

                draw_lineplot(self.results, x_values=eval_steps, x_label='Steps',
                              y_label='Error rates', title='Error rates evolution with training',
                              output_path=(self.evaluation_output_dir / 'error_rates_plot.png'), show=False)

                draw_lineplot({'loss': eval_losses}, x_values=eval_steps, x_label='Steps',
                              y_label='Loss', title='Loss evolution with training',
                              output_path=(self.evaluation_output_dir / 'loss_plot.png'), show=False)

            if step % self.snapshot_rate < self.num_workers and (self.device == 0 or not self.is_distributed):
                self.save_snapshot(step)

        self.evaluate_during_training(step=total_steps)


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    logger.info(f"Process group initialized with {os.environ['WORLD_SIZE']} workers.")


def main(config: dict):
    logger.info(f'Initializing training...')

    # Set random seeds
    random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])

    # Initialize distributed training
    is_distributed = config['num_workers'] > 1

    if is_distributed:
        ddp_setup()

    # Get the model and the number of steps run
    model = OcrTorchModel(config)
    total_steps_run = 0
    if config['snapshot_path'].exists():
        logger.info(f'Loading snapshot from {config["snapshot_path"]}')
        if config['num_workers'] > 1:
            snapshot = torch.load(config['snapshot_path'], map_location='cuda:0')
        else:
            snapshot = torch.load(config['snapshot_path'])
        model.load_state_dict(snapshot['MODEL_STATE'])
        total_steps_run = snapshot['TOTAL_STEPS_RUN']

    per_worker_steps_run = total_steps_run // config['num_workers']

    # Get the datasets and the dataloaders
    train_dataset = OcrIterDataset(data_dir=config['train_data_dir'],
                                   classes=config['classes'],
                                   max_batch_size=config['max_batch_size'],
                                   img_height=config['chunk_height'],
                                   chunk_width=config['chunk_width'],
                                   chunk_overlap=config['chunk_overlap'],
                                   indices_to_classes=config['indices_to_classes'],
                                   classes_to_indices=config['classes_to_indices'],
                                   num_workers=config['num_workers'],
                                   per_worker_steps_run=per_worker_steps_run)

    eval_dataset = OcrIterDataset(data_dir=config['test_data_dir'],
                                  classes=config['classes'],
                                  max_batch_size=config['max_batch_size'],
                                  img_height=config['chunk_height'],
                                  chunk_width=config['chunk_width'],
                                  chunk_overlap=config['chunk_overlap'],
                                  indices_to_classes=config['indices_to_classes'],
                                  classes_to_indices=config['classes_to_indices'],
                                  num_workers=1,  # We are only evaluating on one GPU
                                  per_worker_steps_run=per_worker_steps_run)

    train_dataloader = get_custom_dataloader(train_dataset)
    eval_dataloader = get_custom_dataloader(eval_dataset)

    # Get the optimizer and the scheduler
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduler_gamma'], patience=config['scheduler_patience'], min_lr=0.00001)

    # Initialize the trainer and start training
    trainer = OcrModelTrainer(model=model,
                              train_dataloader=train_dataloader,
                              eval_dataloader=eval_dataloader,
                              per_worker_steps_run=per_worker_steps_run,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              scheduler_step_rate=config['scheduler_step_rate'],
                              snapshot_rate=config['snapshot_rate'],
                              snapshot_path=config['snapshot_path'],
                              evaluation_output_dir=config['evaluation_output_dir'],
                              evaluation_rate=config['evaluation_rate'],
                              chunk_overlap=config['chunk_overlap'],
                              device=config['device'],
                              num_workers=config['num_workers'])

    trainer.train(total_steps=config['total_steps'])

    # Clean up distributed training
    if is_distributed:
        destroy_process_group()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Path to the config file', required=False)
    parser.add_argument('--debug', action='store_true', help='Run in debug mode', required=False, default=False)
    args = parser.parse_args()

    # Get the config
    if args.config_path is not None and not args.debug:
        config = get_config(Path(args.config_path))

    else:
        from ajmc.commons.variables import PACKAGE_DIR

        config = get_config(PACKAGE_DIR / 'tests/test_ocr/config_first_round.json')
        config['device'] = 'cuda'

    # Initialize logging
    config['evaluation_output_dir'].mkdir(parents=True, exist_ok=True)
    logger.info(f"Running with config: {args.config_path}")
    main(config)
