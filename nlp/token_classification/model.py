from typing import Dict, Union, List
import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader


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


