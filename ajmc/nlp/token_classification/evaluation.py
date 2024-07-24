import os
from pathlib import Path
from typing import List, Union, Dict, Optional

import evaluate
import numpy as np
import pandas as pd
import torch
import transformers
from hipe_commons.helpers.tsv import get_tsv_data
from torch.utils.data import DataLoader, SequentialSampler

from ajmc.commons.docstrings import docstrings, docstring_formatter
from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.nlp.token_classification.data_preparation.utils import write_predictions_to_tsv
from ajmc.nlp.token_classification.model import predict, predict_dataset

logger = get_ajmc_logger(__name__)


def seqeval_evaluation(predictions: List[List[str]],
                       groundtruth: List[List[str]],
                       nerify_labels: bool = False):
    """Simple wrapper around seqeval."""
    metric = evaluate.load('seqeval')
    if not nerify_labels:
        return metric.compute(predictions=predictions, references=groundtruth, zero_division=0)
    else:
        return metric.compute(predictions=['B-' + l if l != 'O' else l for l in predictions],
                              references=['B-' + l if l != 'O' else l for l in groundtruth],
                              zero_division=0)


@docstring_formatter(**docstrings)
def evaluate_dataset(dataset: torch.utils.data.Dataset,
                     model: transformers.PreTrainedModel,
                     batch_size: int,
                     ids_to_labels: Dict[int, str],
                     rebuild_ner_labels: bool = True,
                     do_debug: bool = False):
    """Evaluate an entire dataset using seqeval. Is used during the main train loop.

    Args:
        dataset: {custom_dataset}
        model: Self explanatory
        batch_size: Self explanatory
        ids_to_labels: {ids_to_labels}
        rebuild_ner_labels: Whether to simulate NER Labels (for seqeval)
        do_debug: {do_debug}
    """

    if rebuild_ner_labels:
        if not all([l.startswith('B-') or l.startswith('I-') or l == 'O' for l in ids_to_labels.values()]):
            ids_to_labels = {i: 'B-' + l for i, l in ids_to_labels.items()}

    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    logger.info('Running evaluation')
    predictions = None

    for batch in dataloader:
        if predictions is None:
            predictions = predict(batch, model)
            groundtruth = batch['labels'].numpy()
        else:
            predictions = np.append(predictions, predict(batch, model), axis=0)
            groundtruth = np.append(groundtruth, batch['labels'].numpy(), axis=0)
        if do_debug:
            break

    # Remove ignored index (special tokens)

    predictions = [
        [ids_to_labels[p] for (p, l) in zip(example_pred, example_gt) if l != -100]
        for example_pred, example_gt in zip(predictions, groundtruth)
    ]
    groundtruth = [
        [ids_to_labels[l] for l in label if l != -100]
        for label in groundtruth
    ]

    return seqeval_evaluation(predictions, groundtruth)


def evaluate_iob_files(output_dir: Path,
                       groundtruth_path: Path,
                       preds_path: Path,
                       method: str,
                       hipe_script_path: Optional[Path] = None,
                       output_suffix: str = None,
                       task: str = 'nerc_coarse'):
    """Evaluates CLEF-HIPE compliant files.
     If ``method`` is set to ``'hipe'``, runs run CLEF-HIPE-evaluation within ``os.system``. Else if ``method`` is set to
     ``'seqeval``, imports the files as dfs."""

    if method == 'hipe':
        os.system(
                f"""
            python {str(hipe_script_path)} \
            --skip-check \
            --ref {str(groundtruth_path)} \
            --pred {str(preds_path)} \
            --task {task} \
            --outdir {str(output_dir)}
            """
        )

    elif method == 'seqeval':

        preds = pd.read_csv(preds_path.open('r'), delimiter='\t', skiprows=1, comment='#', usecols=[0, 1], names=['TOKEN', 'NE-COARSE-LIT'])
        gt = pd.read_csv(groundtruth_path.open('r'), delimiter="\t", skiprows=1, comment="#", usecols=[0, 1], names=["TOKEN", "NE-COARSE-LIT"])

        # Evaluate with seqeval
        metric = evaluate.load("seqeval")
        results = metric.compute(predictions=[preds["NE-COARSE-LIT"].tolist()],
                                 references=[gt["NE-COARSE-LIT"].tolist()])

        results = seqeval_to_df(results)

        if output_suffix:
            results.to_csv(output_dir / f'{method}_results_{output_suffix}.tsv', sep='\t', index=False)
        else:
            results.to_csv(output_dir / f'{method}_results.tsv', sep='\t', index=False)


def seqeval_to_df(seqeval_output: Dict[str, Union[Dict[str, float], float]],
                  do_debug: bool = False) -> pd.DataFrame:
    """Transforms ``seqeval_output`` to a MultiIndex ``pd.DataFrame``.

    Args:
        seqeval_output: A dict containing:

                        * A dict of metrics for each entity type
                        * A pair ``{'overall_metric': value}`` for each overall metric.

                        Looks like ``{'ent_type1': {'precision':float, 'recall':float}, ... , 'overall_recall':float,...}``.

        do_debug: Fills empty entity types with 0.

    Returns:
        A ``pd.DataFrame`` with a MultiIndex. The first level is the entity type, the second level is the metric.
    """

    abbreviations = {"precision": "P", "recall": "R", "f1": "F1", "accuracy": "A", "number": "N"}
    to_df = {}
    for key in seqeval_output.keys():
        if key.startswith("overall"):

            to_df[("ALL", abbreviations[key.split("_")[1]])] = [seqeval_output[key]]
        else:
            for subkey in seqeval_output[key].keys():
                to_df[(key, abbreviations[subkey])] = [seqeval_output[key][subkey]]

    ordered_keys = [("ALL", key) for key in ["F1", "A", "P", "R"]] + [(k[0], metric) for k in to_df.keys() if k[0] != 'ALL' for metric in
                                                                      ["F1", "P", "R", 'N']]

    if do_debug:
        to_df_debug = {}
        for key in ordered_keys:
            try:
                to_df_debug[key] = to_df[key]
            except KeyError:
                to_df_debug[key] = 0
        return pd.DataFrame(to_df_debug)

    else:
        return pd.DataFrame({key: to_df[key] for key in ordered_keys})


def evaluate_hipe(dataset: 'token_classification.data_preparation.HipeDataset',
                  model: transformers.PreTrainedModel,
                  device: torch.device,
                  ids_to_labels: Dict[int, str],
                  output_dir: Path,
                  labels_column: str,
                  hipe_script_path: Path,
                  groundtruth_tsv_path: Optional[Path] = None,
                  groundtruth_tsv_url: Optional[str] = None,
                  batch_size: int = 8,
                  do_debug: bool = False):
    """Performs the entire pipeline to hipe-evaluate a model, i.e. :
        - Getting the model's prediction on a dataset
        - Reconstructing a HIPE-compliant tsv with these prediction
        - Comparing the predictions-tsv and the corresponding groundtruth-tsv using the hipe scorer.
        - Writing the files.


    """
    # Write tsv locally
    if groundtruth_tsv_url:
        logger.info(f'Downloading a local copy from {groundtruth_tsv_url}')
        groundtruth_tsv_path = output_dir / 'groundtruth.tsv'
        groundtruth_tsv_data = get_tsv_data(url=groundtruth_tsv_url)
        groundtruth_tsv_path.write_text(groundtruth_tsv_data, encoding='utf-8')

    predictions = predict_dataset(dataset=dataset, model=model, do_debug=do_debug).tolist()

    # get the labels, append None if token has no line number
    predictions = [
        [ids_to_labels[p] if l else None for (p, l) in zip(prediction, line_numbers)]
        for prediction, line_numbers in zip(predictions, dataset.tsv_line_numbers)
    ]

    preds_path = output_dir / 'results/hipe_eval/predictions.tsv'
    write_predictions_to_tsv(dataset.words, predictions, dataset.tsv_line_numbers,
                             preds_path,
                             labels_column, groundtruth_tsv_path, groundtruth_tsv_url)

    evaluate_iob_files(output_dir=output_dir / 'results/hipe_eval',
                       groundtruth_path=groundtruth_tsv_path,
                       preds_path=preds_path,
                       method='hipe',
                       hipe_script_path=hipe_script_path)
