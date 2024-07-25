import csv
import unicodedata
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
import transformers
from hipe_commons.helpers.tsv import tsv_to_dict

from ajmc.commons.docstrings import docstrings, docstring_formatter
from ajmc.commons.miscellaneous import get_unique_elements
from ajmc.nlp.token_classification.data_preparation.utils import sort_ner_labels, align_labels_to_tokenized, align_to_tokenized


class HipeDataset(torch.utils.data.Dataset):

    @docstring_formatter(batch_encoding=docstrings['BatchEncoding'])
    def __init__(self,
                 batch_encoding: "BatchEncoding",
                 tsv_line_numbers: List[List[int]],
                 words: List[List[str]],
                 labels: Optional[List[List[int]]] = None,
                 word_ids: Optional[List[List[Union[int, None]]]] = None):
        """Default constructor.

        Args:
            batch_encoding: {batch_encoding}
        """

        self.batch_encoding = batch_encoding
        self.labels = labels
        self.tsv_line_numbers = tsv_line_numbers
        self.words = words
        if word_ids is None:
            self.token_offsets = [e.word_ids for e in batch_encoding.encodings]
        else:
            self.token_offsets = word_ids


    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.batch_encoding.items() if k not in ['overflow_to_sample_mapping', 'word_ids']}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.tsv_line_numbers)


def read_lemlink_tsv(tsv_path: Path) -> pd.DataFrame:
    """Read a TSV file with the format of the lemlink dataset.

    Note:
        THIS IS INTENDED TO READ UNCOMMENTED TSV FILES.
    """

    df = pd.read_csv(tsv_path,
                     # comment='#',
                     sep='\t',
                     header=None,
                     quoting=csv.QUOTE_NONE,
                     dtype=str,
                     keep_default_na=False,
                     names=['TOKEN', 'LABEL', 'ANCHOR_TEXT', 'ANCHOR_TARGET', 'RENDER', 'SEG', 'DOCUMENT_ID'])
    # Add a column with the line number
    df['n'] = df.index
    # Replace the Labels B-Note and I-Note with O
    df['LABEL'] = df['LABEL'].apply(lambda x: 'O' if x in ['B-note', 'I-note'] else x)
    return df



def prepare_datasets(config: 'argparse.Namespace', tokenizer):
    data = {}
    for split in ['train', 'eval']:
        if getattr(config, split + '_path') or getattr(config, split + '_url'):
            if config.data_format == 'ner':
                data[split] = tsv_to_dict(path=getattr(config, split + '_path'), url=getattr(config, split + '_url'))
            elif config.data_format == 'lemlink':
                df = read_lemlink_tsv(getattr(config, split + '_path'))
                data[split] = df.to_dict(orient='list')

    config.unique_labels = sort_ner_labels(
            get_unique_elements([data[k][config.labels_column] for k in data.keys()]))
    config.labels_to_ids = {label: i for i, label in enumerate(config.unique_labels)}
    config.ids_to_labels = {id: tag for tag, id in config.labels_to_ids.items()}
    config.num_labels = len(config.unique_labels)

    if tokenizer.is_fast:
        for split in data.keys():
            data[split]['batchencoding'] = tokenizer(data[split]['TOKEN'],
                                                     padding=True,
                                                     truncation=True,
                                                     max_length=config.model_max_length,
                                                     is_split_into_words=True,
                                                     return_overflowing_tokens=True)

            data[split]['labels'] = [align_labels_to_tokenized(e.word_ids, data[split][config.labels_column], config.labels_to_ids)
                                     for e in data[split]['batchencoding'].encodings]

            data[split]['words'] = [align_to_tokenized(e.word_ids, data[split]['TOKEN']) for e in
                                    data[split]['batchencoding'].encodings]

            data[split]['tsv_line_numbers'] = [align_to_tokenized(e.word_ids, data[split]['n']) for e in
                                               data[split]['batchencoding'].encodings]



    else:
        for split in data.keys():
            data[split]['batchencoding'] = slow_tokenization(data[split]['TOKEN'], tokenizer, config.model_max_length)

            data[split]['labels'] = [align_labels_to_tokenized(word_ids, data[split][config.labels_column], config.labels_to_ids)
                                     for word_ids in data[split]['batchencoding']['word_ids']]

            data[split]['words'] = [align_to_tokenized(word_ids, data[split]['TOKEN']) for word_ids in
                                    data[split]['batchencoding']['word_ids']]

            data[split]['tsv_line_numbers'] = [align_to_tokenized(word_ids, data[split]['n']) for word_ids in
                                               data[split]['batchencoding']['word_ids']]

    datasets = {}
    for split in data.keys():
        datasets[split] = HipeDataset(batch_encoding=data[split]['batchencoding'],
                                      tsv_line_numbers=data[split]["tsv_line_numbers"],
                                      words=data[split]['words'],
                                      labels=data[split]['labels'],
                                      word_ids=None if tokenizer.is_fast else data[split]['batchencoding']['word_ids'])

    return datasets


def create_prediction_dataset(tokenizer, path: Optional[Path] = None, url: Optional[str] = None):
    """Creates a ``HipeDataset`` for prediction (i.e. without labels) from a ``path`` or an ``url``."""
    data = tsv_to_dict(path=str(path), url=url)
    data['batchencoding'] = tokenizer(data['TOKEN'],
                                      padding=True,
                                      truncation=True,
                                      is_split_into_words=True,
                                      return_overflowing_tokens=True)

    data['words'] = [align_to_tokenized(e.word_ids, data['TOKEN']) for e in
                     data['batchencoding'].encodings]

    data['tsv_line_numbers'] = [align_to_tokenized(e.word_ids, data['n']) for e in
                                data['batchencoding'].encodings]

    data['labels'] = [[0 for _ in x] for x in data['tsv_line_numbers']]  # dummy labels without which I get an untracable error from HF

    return HipeDataset(batch_encoding=data['batchencoding'],
                       tsv_line_numbers=data["tsv_line_numbers"],
                       words=data['words'],
                       labels=data['labels'])


def slow_tokenization(inputs: List[str], tokenizer: transformers.PreTrainedTokenizer, model_max_length: int,
                      add_special_tokens: bool = False) -> dict:
    """Use this function to tokenize a list of strings with a given "slow" tokenizer (as opposed to HF's ``TokenizerFast``) and model_max_length."""

    inputs = ' '.join(inputs)
    inputs = unicodedata.normalize('NFC', inputs)  # For CANINE

    batchencoding = tokenizer(inputs,
                              padding=True,
                              truncation=False,
                              pad_to_multiple_of=model_max_length - 2,
                              add_special_tokens=False)

    # get the word_ids
    word_ids = []
    word_id = 0
    for c in inputs:
        if c == ' ':
            word_id += 1
            word_ids.append(None)
        else:
            word_ids.append(word_id)

    word_ids += [None] * (len(batchencoding['input_ids']) - len(word_ids))

    custom_encoding = {k: [] for k in batchencoding.keys()}
    custom_encoding['word_ids'] = []

    for i in range(0, len(word_ids), model_max_length - 2):
        for k, v in batchencoding.items():
            if add_special_tokens:
                custom_encoding[k].append([tokenizer.bos_token_id] + v[i:i + model_max_length - 2] + [tokenizer.eos_token_id])
            else:
                custom_encoding[k].append(v[i:i + model_max_length - 2])

        if add_special_tokens:
            custom_encoding['word_ids'].append([None] + word_ids[i:i + model_max_length - 2] + [None])
        else:
            custom_encoding['word_ids'].append(word_ids[i:i + model_max_length - 2])

    return custom_encoding
