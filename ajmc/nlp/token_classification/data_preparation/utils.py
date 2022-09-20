from typing import List, Dict, Union, Any, Iterable

import torch
from hipe_commons.helpers.tsv import get_tsv_data
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.miscellaneous import get_custom_logger, split_list

logger = get_custom_logger(__name__)


def nerify_labels(labels: Iterable[str],
                  add_i: bool = True,
                  add_o: bool = True) -> List[str]:
    """A simple function to NERify labels, adding `'B-'`s, `'I-'`s and `'O'` labels, making them CoLLN-compliant.

    Args:
        labels: The iterable of unique labels to NERify.
        add_i: If false, does not add `'I-'` labels.
        add_o: If false, does not add `'O'` label.

    Returns:
        The list of nerified labels
    """
    ner_labels = ['B-' + l for l in labels if l != 'O']
    if not add_i:
        ner_labels += ['I-' + l for l in labels if l != 'O']
    if 'O' in labels or add_o:
        ner_labels.append('O')

    return sort_ner_labels(ner_labels)


def sort_ner_labels(labels: Iterable[str]):
    """Sorts a list of CoLLN-compliant labels alphabetically, ending with 'O'.

    Args:
        labels: The iterable of unique labels to sort. Each label should be in the form
                `'B-classname'`, `'I-classname'` or `'O'`.

    Returns:
        The sorted list of labels
    """

    sorted_labels = sorted([l for l in labels if l != 'O'], key=lambda x: x[2:] + x[0])
    if 'O' in labels:
        sorted_labels.append('O')

    return sorted_labels


def align_labels_to_tokenized(tokens_to_words_offsets: List[Union[None, int]],
                              labels: List[str],
                              labels_to_ids: Dict[str, int],
                              label_all_tokens: bool = False,
                              null_label: object = -100) -> List[int]:
    """`align_labels_to_tokenized` is a special case of `align_to_tokenized`, dealing with labels specificities.

    As such, it will:
        - Change `labels` to their corresponding ids
        - Label all the sub-tokens (with a single `'B-'` label) if `label_all_tokens` is True.
        - Append `null_label` instead of `None` if the token offset is None.
    """



    previous_token_index = None
    aligned_labels = []

    for token_index in tokens_to_words_offsets:
        if token_index is None:
            aligned_labels.append(null_label)

        elif token_index != previous_token_index:
            aligned_labels.append(labels_to_ids[labels[token_index]])

        else:
            if not label_all_tokens:
                aligned_labels.append(null_label)
            else:
                aligned_labels.append(
                    labels_to_ids['I' + labels[token_index][1:] if labels[token_index] != 'O' else 'O'])

        previous_token_index = token_index

    return aligned_labels


def align_to_tokenized(tokens_to_words_offsets: List[Union[None, int]],
                       to_align: List[Any]) -> List[Any]:
    """Align `to_align` to a list of offsets, appending `None` if the offset is None.

    This is used to align a list of elements to their tokenized equivalent, for instance to align words
    to tokens. Example :

        ```python
        words = ['Hello', 'world']
        tokens = ['he', '#llo', 'w', '#o', '#rld']
        # aligned words would be ['Hello', None, 'world', None, None]
        ```
    """

    previous_token_index = None
    aligned_elements = []

    for token_index in tokens_to_words_offsets:
        if token_index is None:
            aligned_elements.append(None)

        elif token_index != previous_token_index:
            aligned_elements.append(to_align[token_index])

        else:
            aligned_elements.append(None)
        previous_token_index = token_index

    return aligned_elements


def align_from_tokenized(tokens_to_words_offsets: List[Union[None, int]],
                         to_align: List[object]) -> List[object]:
    """Returns the elements of `to_align` if the corresponding element in `tokens_to_words_offsets` is not and is
    different from the previous one.

    This is used to align a list of tokenized elements to their untokenized equivalent, for instance to align labels
    to word. It does the contrary to `align_to_tokenized`. Example :

        ```python
        tokens = ['he', '#llo', 'w', '#o', '#rld']
        offsets= [ 0,    0,      1,   1,    1    ]
        labels = [ 0,    1,      2,   3,    2    ]
        words =  ['Hello',      'world'          ]

        # aligned labels would be [0, 2]
        ```
    """
    previous_token_index = None
    aligned_elements = []

    for i, token_index in enumerate(tokens_to_words_offsets):
        if token_index is None:
            continue
        if token_index != previous_token_index:
            aligned_elements.append(to_align[i])

        previous_token_index = token_index

    return aligned_elements


# LEGACY. To use with HIPE.
def write_predictions_to_tsv(words: List[List[Union[str, None]]],
                             labels: List[List[Union[str, None]]],
                             tsv_line_numbers: List[List[Union[int, None]]],
                             output_file: str,
                             labels_column: str,
                             tsv_path: str = None,
                             tsv_url: str = None, ):
    """Get the source tsv, replaces its labels with predicted labels and write a new file to `output`.

    `words`, `labels` and `tsv_line_numbers` should be three alined list, so as in HipeDataset.
    """

    logger.info(f'Writing predictions to {output_file}')

    tsv_lines = [l.split('\t') for l in get_tsv_data(tsv_path, tsv_url).split('\n')]
    label_col_number = tsv_lines[0].index(labels_column)
    for i in range(len(words)):
        for j in range(len(words[i])):
            if words[i][j]:
                assert tsv_lines[tsv_line_numbers[i][j]][0] == words[i][j]
                tsv_lines[tsv_line_numbers[i][j]][label_col_number] = labels[i][j]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(['\t'.join(l) for l in tsv_lines]))


# LEGACY.
@docstring_formatter(max_length=docstrings['max_length'], special_tokens=docstrings['special_tokens'])
def manual_truncation(tokens, inputs: Dict[str, list], special_tokens: Dict[str, Dict[str, Any]], max_length):
    """Manually truncates and pads model inputs.

    Args:
        inputs: Dict-like outputs of the tokenizer, containing the **untruncated** model's features (e.g.
            'inputs_ids', 'attention_mask'...).
        special_tokens: {special_tokens}
        max_length: {max_length}
    """

    # LEGACY. FOR LAYOUTLM. Note : special tokens are attributes of the tokenizer, find them there.
    # special_tokens = {
    #     'start': {'input_ids': 101, 'bbox': [0, 0, 0, 0], 'token_type_ids': 0, 'labels': -100, 'attention_mask': 1},
    #     'end': {'input_ids': 102, 'bbox': [1000, 1000, 1000, 1000], 'token_type_ids': 0, 'labels': -100,
    #             'attention_mask': 1},
    #     'pad': {'input_ids': 0, 'bbox': [0, 0, 0, 0], 'token_type_ids': 0, 'labels': -100, 'attention_mask': 0},
    # }

    for k in inputs.keys():
        inputs[k] = inputs[k][1:-1]  # We start by triming the first and last tokens
        tokens[k] = split_list(inputs[k], max_length - 2,
                               special_tokens['pad'][k])  # We divide our list in lists of len 510
        inputs[k] = [[special_tokens['start'][k]] + ex + [special_tokens['end'][k]] for ex in inputs[k]]

    return inputs


class CustomDataset(torch.utils.data.Dataset):
    """A custom dataset to wrap a transformers `BatchEncoding` resulting from a TokenizerFast, i.e. """

    @docstring_formatter(**docstrings)
    def __init__(self,
                 encodings: 'BatchEncoding',
                 model_inputs_names: List[str]):
        """Default constructor.

        Args:
            encodings: {BatchEncoding}
            model_inputs_names: {transformers_model_inputs_names}
        """
        assert encodings.encodings is not None, """The provided `BatchEncoding` as `self.encodings` set to `None`. 
        Please use a `TokenizerFast` to avoid this."""

        self.encodings = encodings
        self.model_inputs = model_inputs_names

    def __getitem__(self, idx):
        item = {k: torch.tensor(self.encodings[k][idx]) for k in self.model_inputs}
        if 'labels' in self.encodings.keys():
            item['labels'] = torch.tensor(self.encodings['labels'][idx])

        return item

    def __len__(self):
        return len(self.encodings.encodings)
