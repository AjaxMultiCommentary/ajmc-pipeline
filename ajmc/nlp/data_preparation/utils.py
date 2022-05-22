from typing import Iterable, List, Dict, Union

from ajmc.commons.miscellaneous import get_custom_logger

logger = get_custom_logger(__name__)

def sort_ner_labels(labels: Iterable[str]):
    """Sorts a list of CoLLN-compliant labels alphabetically, starting 'O'."""

    assert 'O' in labels, """Label 'O' not found in labels."""
    labels = [l for l in labels if l != 'O']
    uniques = set([l[2:] for l in labels])

    sorted_labels = ['O']
    for l in sorted(uniques):
        sorted_labels.append('B-' + l)
        sorted_labels.append('I-' + l)

    return sorted_labels


def align_labels(tokens_to_words_offsets: 'transformers.tokenizers.Encoding',
                 labels: List[str],
                 labels_to_ids: Dict[str, int],
                 label_all_tokens: bool = False,
                 null_label: object = -100) -> List[List[int]]:
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


def align_elements(tokens_to_words_offsets: 'transformers.tokenizers.Encoding',
                   elements: List[object]) -> List[object]:
    """Align `elements` to a list of offsets, appending `None` if the offset is None."""

    previous_token_index = None
    aligned_elements = []

    for token_index in tokens_to_words_offsets:
        if token_index is None:
            aligned_elements.append(None)

        elif token_index != previous_token_index:
            aligned_elements.append(elements[token_index])

        else:
            aligned_elements.append(None)
        previous_token_index = token_index

    return aligned_elements


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