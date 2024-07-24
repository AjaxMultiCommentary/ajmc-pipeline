"""This module is highly inspired by HuggingFace's
`run_ner.py <https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py>`_.
It runs on a single GPU."""
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import safetensors
import torch
import transformers
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from ajmc.commons.miscellaneous import get_ajmc_logger, ROOT_LOGGER
from ajmc.nlp.token_classification.config import AjmcNlpConfig
from ajmc.nlp.token_classification.data_preparation.hipe_iob import prepare_datasets
from ajmc.nlp.token_classification.evaluation import evaluate_dataset, seqeval_to_df, evaluate_hipe
from ajmc.nlp.token_classification.model import predict_and_write_tsv

logger = get_ajmc_logger(__name__)


def set_seed(seed):
    """Sets seed for ``random``, ``np.random`` and ``torch``."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_canine_state_dict_from_shiba_safetensors(safetensors_path: str) -> dict:
    with safetensors.safe_open(safetensors_path, framework='pt') as model_file:
        state_dict = {}
        for k in model_file.keys():
            if k.startswith('shiba_model.'):
                state_dict[k[12:]] = model_file.get_tensor(k)

    return state_dict


def train(config: AjmcNlpConfig,
          model: transformers.PreTrainedModel,
          train_dataset: 'token_classification.data_preparation.HipeDataset',
          eval_dataset: 'token_classification.data_preparation.HipeDataset',
          tokenizer: transformers.PreTrainedTokenizer):
    """Main training function.

    This function does the following:

    * Does the training on ``train_dataset``
    * At the end of each epoch:

      * evaluate the model on ``eval_dataset`` using seqeval
      * saves a model checkpoint
      * saves the model as best_model if model has highest scores

    Args:
        config: An ``argparse.Namespace`` containing the required ``transformers.TrainingArguments``
        model: A transformers model instance
        train_dataset: The ``torch.utils.data.Dataset`` on which to train
        eval_dataset: The ``torch.utils.data.Dataset`` on which to evaluate the model
        tokenizer: The model's tokenizer.
    """
    model.to(config.device)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)

    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total
    )
    # =================== Pretraining declarations ====================
    logger.info(f"""Running training on {len(train_dataset)} examples, for {config.epochs} epochs.""")

    global_step = 0
    best_f1 = -1
    count_no_improvement = 0
    train_results = pd.DataFrame()

    model.zero_grad()
    set_seed(config.seed)

    for epoch_num in range(config.epochs):

        logger.info(f"Starting epoch {epoch_num}")

        loss_batches_list = []
        epoch_time = time.time()
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training ep. {epoch_num}"):

            inputs = {key: batch[key].to(config.device) for key in batch.keys()}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are tuples in pytorch and transformers

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            loss_batches_list.append(loss.item())

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if config.do_debug:
                    break

        # ============================ Evaluate and append during training ============================
        if config.evaluate_during_training:
            model.eval()
            epoch_results = evaluate_dataset(dataset=eval_dataset,
                                             model=model,
                                             batch_size=config.batch_size,
                                             ids_to_labels=config.ids_to_labels,
                                             do_debug=config.do_debug)
            epoch_results = seqeval_to_df(epoch_results)

            epoch_data = pd.DataFrame({("TRAINING", "EP"): [epoch_num + 1],
                                       ("TRAINING", "TIME"): [time.time() - epoch_time],
                                       ("TRAINING", "LOSS"): [np.mean(loss_batches_list)]})
            epoch_results = pd.concat([epoch_data, epoch_results], axis=1)
            train_results = pd.concat([train_results, epoch_results], axis=0, ignore_index=True)

            # ========================= Save best model and write its results ===============================
            if round(epoch_results[('ALL', 'F1')][0], 4) > round(best_f1, 4):

                count_no_improvement = 0
                best_f1 = epoch_results[('ALL', 'F1')][0]
                epoch_results.to_csv(config.seqeval_output_dir / "best_results.tsv", sep='\t', index=False)

                if config.do_save:
                    model.save_pretrained(config.model_save_dir)
                    tokenizer.save_pretrained(config.model_save_dir)
                    torch.save(model.state_dict(), config.model_save_dir / "training_args.bin")

            else:
                count_no_improvement += 1

            if count_no_improvement == config.early_stopping_patience and config.do_early_stopping:
                break

        if config.do_debug:
            break
    if config.evaluate_during_training:
        train_results.to_csv(config.seqeval_output_dir / 'train_results.tsv', sep='\t', index=False)

    elif config.do_save:
        model.save_pretrained(config.model_save_dir)
        tokenizer.save_pretrained(config.model_save_dir)
        torch.save(config, config.model_save_dir / "training_args.bin")


def create_dirs(config: AjmcNlpConfig):
    # Create directories
    config.output_dir.mkdir(exist_ok=config.overwrite_outputs, parents=True)
    config.model_save_dir.mkdir(exist_ok=config.overwrite_outputs, parents=True)
    config.predictions_dir.mkdir(exist_ok=config.overwrite_outputs, parents=True)
    config.seqeval_output_dir.mkdir(exist_ok=config.overwrite_outputs, parents=True)
    config.hipe_output_dir.mkdir(exist_ok=config.overwrite_outputs, parents=True)


def main(config_path: Optional[Path] = None, config_dict: Optional[dict] = None):
    """Main entrypoint.

    Args:
        config_path: The path to the JSON file containing the config.
    """

    # Create the config
    if config_path is not None:
        config = AjmcNlpConfig.from_json(config_path)
    elif config_dict is not None:
        config = AjmcNlpConfig.from_dict(config_dict)
    else:
        raise ValueError("Either config_path or config_dict must be provided.")

    ROOT_LOGGER.setLevel("DEBUG" if config.do_debug else "INFO")
    logger.info(f"""Runing pipeline on {config.output_dir.name}""")

    create_dirs(config)

    # Save config
    config.to_json(config.output_dir / 'config.json')

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
    # Todo: move to config
    if 'roberta' in config.model_name or config.model_name == 'bowphs/PhilBerta':
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, add_prefix_space=True)  # for roberta exclusively

    datasets = prepare_datasets(config, tokenizer)

    if not config.model_path:
        model = transformers.AutoModelForTokenClassification.from_pretrained(config.model_name,
                                                                             num_labels=config.num_labels)
    else:
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            config.model_name, num_labels=config.num_labels,
            state_dict=get_canine_state_dict_from_shiba_safetensors(str(config.model_path)))

    if config.do_train:
        train(config=config,
              model=model,
              train_dataset=datasets['train'],
              eval_dataset=datasets['eval'] if config.evaluate_during_training else None,
              tokenizer=tokenizer)

    if config.do_hipe_eval:
        evaluate_hipe(dataset=datasets['eval'],
                      model=model,
                      device=config.device,
                      ids_to_labels=config.ids_to_labels,
                      output_dir=config.hipe_output_dir,
                      labels_column=config.labels_column,
                      hipe_script_path=config.hipe_script_path,
                      groundtruth_tsv_path=config.eval_path,
                      groundtruth_tsv_url=config.eval_url,
                      )

    if config.do_seqeval:
        seq_results = evaluate_dataset(dataset=datasets['eval'],
                                       model=model,
                                       batch_size=config.batch_size,
                                       ids_to_labels=config.ids_to_labels)
        seq_results = seqeval_to_df(seq_results)
        seq_results.to_csv(config.seqeval_output_dir / 'final_results.tsv', sep='\t', index=False)

    if config.do_predict:
        for url in config.predict_urls:
            predict_and_write_tsv(model=model, output_dir=config.predictions_dir,
                                  tokenizer=tokenizer, ids_to_labels=config.ids_to_labels,
                                  labels_column=config.labels_column, url=url)

        for path in config.predict_paths:
            predict_and_write_tsv(model=model, output_dir=config.predictions_dir,
                                  tokenizer=tokenizer, ids_to_labels=config.ids_to_labels,
                                  labels_column=config.labels_column, path=path)


if __name__ == '__main__':
    from ajmc.commons import variables as vs

    # Todo ‼️ : change the path to the config file
    main(config_path=vs.PACKAGE_DIR / 'configs/token_classification/ajmc_lemlink_test.json')
