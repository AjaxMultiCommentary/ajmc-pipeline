import os
from typing import Dict, Union, List, Optional
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, SequentialSampler

from ajmc.nlp.data_preparation.hipe_iob import create_prediction_dataset
from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.nlp.data_preparation.utils import write_predictions_to_tsv

logger = get_custom_logger(__name__)

def predict(inputs: Dict[str, torch.tensor], model: 'transformers.models', device: torch.device):
    """Predicts for a batch or a single example.

    :param inputs: a mapping to between the names of the model's requirements and a tensor containing example(s).
    Example : `{'input_ids': torch.tensor([[int, int, ...], [int, int, ...]])`.

    :returns: a np.ndarray containing the predicted labels, so in the shape (number of exs, length of an ex). """

    model.eval()

    with torch.no_grad():
        inputs = {key: inputs[key].to(device) for key in inputs.keys()}
        outputs = model(**inputs)
        return np.argmax(outputs[1].detach().cpu().numpy(), axis=2)



# Todo I think this should run directly on a dataset
def predict_batches(batches: Union[torch.utils.data.dataloader.DataLoader, List[Dict[str, torch.tensor]]],
                    model: transformers.PreTrainedModel,
                    device: torch.device,
                    do_debug: bool = False) -> np.ndarray:
    """Runs `predict` on a collection of batches."""

    predictions = None

    for batch in batches:
        if predictions is None:
            predictions = predict(batch, model, device)
        else:
            predictions = np.append(predictions, predict(batch, model, device), axis=0)

        if do_debug:
            break

    return predictions


def predict_dataset(dataset: 'HipeDataset',
                    model: transformers.PreTrainedModel,
                    device: torch.device,
                    do_debug: bool = False,
                    batch_size: int = 8,
                    ) -> np.ndarray:
    """Runs `predict` on a dataset."""

    dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

    return predict_batches(dataloader, model, device=device, do_debug=do_debug)


def predict_and_write_tsv(model, device, output_dir, tokenizer, ids_to_labels, labels_column: str,
                          path: Optional[str] = None, url: Optional[str] = None):
    """Creates a dataset from a tsv, predicts and writes predictions to tsv."""

    logger.info(f"""Starting prediction on {path.split('/')[-1] if path else url.split('/')[-1]}""")
    dataset_to_pred = create_prediction_dataset(tokenizer=tokenizer, path=path, url=url)
    predictions = predict_dataset(dataset_to_pred, model, device)

    predictions = [
        [ids_to_labels[p] if l else None for (p, l) in zip(prediction, line_numbers)]
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