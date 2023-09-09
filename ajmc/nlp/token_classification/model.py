import os
from typing import Dict, Union, List, Optional

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, DataLoader, SequentialSampler

from ajmc.commons.docstrings import docstrings, docstring_formatter
from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.nlp.token_classification.data_preparation.hipe_iob import create_prediction_dataset
from ajmc.nlp.token_classification.data_preparation.utils import write_predictions_to_tsv

logger = get_ajmc_logger(__name__)


@docstring_formatter(**docstrings)
def predict(model_inputs: Dict[str, torch.tensor],
            model: 'transformers.models'):
    """Predicts for a batch or a single example.

    Args:
        model_inputs: {transformers_model_inputs}
        model: {transformers_model}

    Returns:
        {transformers_model_predictions} """

    model.eval()

    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k,v in model_inputs.items()})

        return np.argmax(outputs['logits'].detach().cpu().numpy(), axis=2)


def predict_batches(batches: Union[torch.utils.data.dataloader.DataLoader, List[Dict[str, torch.tensor]]],
                    model: transformers.PreTrainedModel,
                    do_debug: bool = False) -> np.ndarray:
    """Runs ``predict`` on a collection of batches."""

    predictions = None

    for batch in batches:
        if predictions is None:
            predictions = predict(batch, model)
        else:
            predictions = np.append(predictions, predict(batch, model), axis=0)

        if do_debug:
            break

    return predictions


@docstring_formatter(**docstrings)
def predict_dataset(dataset: torch.utils.data.Dataset,
                    model: transformers.PreTrainedModel,
                    do_debug: bool = False,
                    batch_size: int = 8,
                    ) -> np.ndarray:
    """Runs ``predict`` on a dataset.

    Args:
        dataset: {custom_dataset}
        model: {transformers_model}
        do_debug: {do_debug}
        batch_size: Self explanatory.

    Returns:
        {transformers_model_predictions}
    """

    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    return predict_batches(dataloader, model, do_debug=do_debug)



def predict_and_write_tsv(model, output_dir, tokenizer, ids_to_labels, labels_column: str,
                          path: Optional[str] = None, url: Optional[str] = None):
    """Creates a dataset from a tsv, predicts and writes predictions to tsv."""

    logger.info(f"""Starting prediction on {path.split('/')[-1] if path else url.split('/')[-1]}""")
    dataset_to_pred = create_prediction_dataset(tokenizer=tokenizer, path=path, url=url)
    predictions = predict_dataset(dataset_to_pred, model)

    predictions = [
        [ids_to_labels[p] if l else None for p, l in zip(prediction, line_numbers)]
        for prediction, line_numbers in zip(predictions, dataset_to_pred.tsv_line_numbers)
    ]

    preds_path = os.path.join(output_dir, url.split('/')[-1])

    write_predictions_to_tsv(words=dataset_to_pred.words,
                             labels=predictions,
                             tsv_line_numbers=dataset_to_pred.tsv_line_numbers,
                             output_file=preds_path,
                             labels_column=labels_column,
                             tsv_path=path,
                             tsv_url=url)


