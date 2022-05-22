import pytest
from ajmc.nlp.data_preparation.utils import align_labels, align_elements


def test_align_labels(test_tokens_to_words_offsets,
                      test_labels_to_ids, test_labels):
    aligned = align_labels(tokens_to_words_offsets=test_tokens_to_words_offsets,
                           labels=test_labels,
                           labels_to_ids=test_labels_to_ids)

    assert len(aligned) == len(test_tokens_to_words_offsets)


def test_align_elements(test_tokens_to_words_offsets,
                        test_labels):
    aligned = align_elements(tokens_to_words_offsets=test_tokens_to_words_offsets,
                             elements=test_labels)

    assert len(aligned) == len(test_tokens_to_words_offsets)