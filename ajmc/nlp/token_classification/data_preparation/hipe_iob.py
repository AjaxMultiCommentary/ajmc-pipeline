from hipe_commons.helpers.tsv import tsv_to_dict
from typing import List, Optional
import torch
from ajmc.nlp.token_classification.data_preparation.utils import sort_ner_labels, align_labels, align_elements
from ajmc.commons.miscellaneous import get_unique_elements


class HipeDataset(torch.utils.data.Dataset):

    def __init__(self,
                 batch_encoding: "BatchEncoding",
                 tsv_line_numbers: List[List[int]],
                 words: List[List[str]],
                 labels: Optional[List[List[int]]] = None):
        self.batch_encoding = batch_encoding
        self.labels = labels
        self.tsv_line_numbers = tsv_line_numbers
        self.words = words
        self.token_offsets = [e.word_ids for e in batch_encoding.encodings]

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.batch_encoding.items() if k != 'overflow_to_sample_mapping'}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.tsv_line_numbers)


def prepare_datasets(config: 'argparse.Namespace', tokenizer):
    data = {}
    for split in ['train', 'eval']:
        if config.__dict__[split + '_path'] or config.__dict__[split + '_url']:
            data[split] = tsv_to_dict(path=config.__dict__[split + '_path'], url=config.__dict__[split + '_url'])

    config.unique_labels = sort_ner_labels(
        get_unique_elements([data[k][config.labels_column] for k in data.keys()]))
    config.labels_to_ids = {label: i for i, label in enumerate(config.unique_labels)}
    config.ids_to_labels = {id: tag for tag, id in config.labels_to_ids.items()}
    config.num_labels = len(config.unique_labels)

    for split in data.keys():
        data[split]['batchencoding'] = tokenizer(data[split]['TOKEN'],
                                                 padding=True,
                                                 truncation=True,
                                                 is_split_into_words=True,
                                                 return_overflowing_tokens=True)

        data[split]['labels'] = [align_labels(e.word_ids, data[split][config.labels_column],
                                              config.labels_to_ids)
                                 for e in data[split]['batchencoding'].encodings]

        data[split]['words'] = [align_elements(e.word_ids, data[split]['TOKEN']) for e in
                                data[split]['batchencoding'].encodings]

        data[split]['tsv_line_numbers'] = [align_elements(e.word_ids, data[split]['n']) for e in
                                           data[split]['batchencoding'].encodings]

    datasets = {}
    for split in data.keys():
        datasets[split] = HipeDataset(batch_encoding=data[split]['batchencoding'],
                                      tsv_line_numbers=data[split]["tsv_line_numbers"],
                                      words=data[split]['words'],
                                      labels=data[split]['labels'],)

    return datasets


def create_prediction_dataset(tokenizer, path: Optional[str] = None, url: Optional[str] = None):
    """Creates a `HipeDataset` for prediction (i.e. without labels) from a `path` or an `url`."""
    data = tsv_to_dict(path=path, url=url)
    data['batchencoding'] = tokenizer(data['TOKEN'],
                                      padding=True,
                                      truncation=True,
                                      is_split_into_words=True,
                                      return_overflowing_tokens=True)

    data['words'] = [align_elements(e.word_ids, data['TOKEN']) for e in
                     data['batchencoding'].encodings]

    data['tsv_line_numbers'] = [align_elements(e.word_ids, data['n']) for e in
                                data['batchencoding'].encodings]

    data['labels'] = [[0 for _ in x] for x in data['tsv_line_numbers']]  # dummy labels without which I get an untracable error from HF

    return HipeDataset(batch_encoding=data['batchencoding'],
                       tsv_line_numbers=data["tsv_line_numbers"],
                       words=data['words'],
                       labels = data['labels'])



