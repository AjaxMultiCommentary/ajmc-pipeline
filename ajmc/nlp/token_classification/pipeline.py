"""This module is highly inspired by HuggingFace's 
[`run_ner.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py). 
It runs on a single GPU."""

import json
import logging
import time
import pandas as pd
import os
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, RandomSampler
from ajmc.nlp.token_classification.evaluation import evaluate_dataset, seqeval_to_df, evaluate_hipe
from ajmc.nlp.token_classification.config import initialize_config
from ajmc.nlp.data_preparation.hipe_iob import prepare_datasets
from ajmc.nlp.token_classification.utils import set_seed
from ajmc.commons.miscellaneous import get_custom_logger


def train(config: 'argparse.Namespace',
          model: transformers.PreTrainedModel,
          train_dataset: 'token_classification.data_preparation.HipeDataset',
          eval_dataset: 'token_classification.data_preparation.HipeDataset',
          tokenizer: transformers.PreTrainedTokenizer):
    """
    Main function of the the script :
        - Does the training on `train_dataset`
        - At the end of each epoch :
            - evaluate the model on `eval_dataset` using seqeval
            - saves a model checkpoint
            - saves the model as best_model if model has highest scores

    :param config: An `argparse.Namespace` containing the required `transformers.TrainingArguments`
    :param model: A transformers model instance
    :param train_dataset: The `torch.utils.data.Dataset` on which to train
    :param eval_dataset: The `torch.utils.data.Dataset` on which to evaluate the model
    :param tokenizer: The model's tokenizer.
    """

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
    best_f1 = 0.5
    count_no_improvement = 0
    train_results = pd.DataFrame()

    model.zero_grad()
    set_seed(config.seed)

    for epoch_num in range(config.epochs):

        logger.info(f"Starting epoch {epoch_num}")

        loss_batches_list = []
        epoch_time = time.time()

        for step, batch in enumerate(train_dataloader):

            model.train()
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
        epoch_results = evaluate_dataset(eval_dataset, model, config.batch_size, config.device,
                                                       config.ids_to_labels, config.do_debug)
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

            best_model_dir = os.path.join(config.output_dir, "best_model")
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            torch.save(config, os.path.join(best_model_dir, "training_args.bin"))
            epoch_results.to_csv(os.path.join(config.output_dir, "results/seqeval/best_results.tsv"), sep='\t',
                                 index=False)

        else:
            count_no_improvement += 1

        if (count_no_improvement == config.early_stopping_patience and config.do_early_stopping) or config.do_debug:
            break

    train_results.to_csv(os.path.join(config.output_dir, "results/seqeval/train_results.tsv"), sep='\t', index=False)


def main(config):
    logger.info(f'Runing pipeline on {config.output_dir.split("/")[-1]}')
    # Create directories
    os.makedirs(config.output_dir, exist_ok=config.overwrite_output_dir)
    os.makedirs(os.path.join(config.output_dir, 'best_model'), exist_ok=config.overwrite_output_dir)
    os.makedirs(os.path.join(config.output_dir, 'results/seqeval'), exist_ok=config.overwrite_output_dir)
    os.makedirs(os.path.join(config.output_dir, 'results/hipe_eval'), exist_ok=config.overwrite_output_dir)

    # Save config
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, skipkeys=True, indent=4, sort_keys=True,
                  default=lambda o: '<not serializable>')

    # todo change model_name_or_path to model_config ; make a double path on data
    # tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name_or_path, add_prefix_space=True)  # for roberta exclusively
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name_or_path)

    datasets = prepare_datasets(config, tokenizer)

    model = transformers.AutoModelForTokenClassification.from_pretrained(config.model_name_or_path,
                                                                         num_labels=config.num_labels)
    model.to(config.device)

    if config.do_train:
        train(config, model, datasets['train'], datasets['eval'], tokenizer)

    if config.do_eval:
        evaluate_hipe(dataset=datasets['eval'],
                      model=model,
                      device=config.device,
                      ids_to_labels=config.ids_to_labels,
                      output_dir=config.output_dir,
                      labels_column=config.labels_column,
                      hipe_script_path=config.hipe_script_path,
                      groundtruth_tsv_path=config.eval_path,
                      groundtruth_tsv_url=config.eval_url,
                      )


if __name__ == '__main__':
    logger = get_custom_logger(__name__, level=logging.DEBUG)
    config = initialize_config(
        # json_path='data/ajmc_de_coarse.json'
    )
    main(config)

#%%
# from hipe_commons.helpers.tsv import get_tsv_data
# logger = get_custom_logger(__name__, )
# config = initialize_config(json_path='/scratch/sven/tmp/HIPE-2022-baseline/token_classification/data/ajmc_de_coarse.json')
# tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name_or_path)
#
# data = get_tsv_data(url=config.train_url)
# len(data.split('\n'))
# datasets = prepare_datasets(config, tokenizer)
#
# model = transformers.AutoModelForTokenClassification.from_pretrained(config.model_name_or_path,
#                                                                      num_labels=config.num_labels)
# model.to(config.device)


# todo reimplement freeze, additionnal data and compare with ner for classics



