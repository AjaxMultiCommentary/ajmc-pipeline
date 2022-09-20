from ajmc.nlp.token_classification.data_preparation import align_labels_to_tokenized, align_to_tokenized


def test_align_labels(test_tokens_to_words_offsets,
                      test_labels_to_ids, test_labels):
    aligned = align_labels_to_tokenized(tokens_to_words_offsets=test_tokens_to_words_offsets,
                                        labels=test_labels,
                                        labels_to_ids=test_labels_to_ids)

    assert len(aligned) == len(test_tokens_to_words_offsets)


def test_align_to_tokenized(test_tokens_to_words_offsets,
                        test_labels):
    aligned = align_to_tokenized(tokens_to_words_offsets=test_tokens_to_words_offsets,
                                 to_align=test_labels)

    assert len(aligned) == len(test_tokens_to_words_offsets)