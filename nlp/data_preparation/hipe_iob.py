from hipe_commons.helpers.tsv import tsv_to_dict
from typing import List
import torch
from nlp.data_preparation.utils import sort_ner_labels, align_labels, align_elements
from utils.general_utils import get_unique_elements

class HipeDataset(torch.utils.data.Dataset):

    def __init__(self,
                 batch_encoding: "BatchEncoding",
                 labels: List[List[int]],
                 tsv_line_numbers: List[List[int]],
                 words: List[List[str]]):
        self.batch_encoding = batch_encoding
        self.labels = labels
        self.tsv_line_numbers = tsv_line_numbers
        self.words = words
        self.token_offsets = [e.word_ids for e in batch_encoding.encodings]

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.batch_encoding.items() if k!='overflow_to_sample_mapping'}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Todo : sort the mess with tokenizer call
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

        data[split]['labels'] = [align_labels(e.word_ids, data[split][config.labels_column], config.labels_to_ids)
                                 for e in data[split]['batchencoding'].encodings]

        data[split]['words'] = [align_elements(e.word_ids, data[split]['TOKEN']) for e in
                                data[split]['batchencoding'].encodings]

        data[split]['tsv_line_numbers'] = [align_elements(e.word_ids, data[split]['n']) for e in
                                           data[split]['batchencoding'].encodings]


    datasets = {}
    for split in data.keys():
        datasets[split] = HipeDataset(data[split]['batchencoding'],
                                      data[split]['labels'],
                                      data[split]["tsv_line_numbers"],
                                      data[split]['words'])

    return datasets


