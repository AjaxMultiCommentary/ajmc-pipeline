"""This module contains sample objects which are sent to `sample_objects.json` and used as fixtures elsewhere."""

from ajmc.nlp.token_classification.evaluation import seqeval_evaluation

sample_ner_labels_pred = ['O', 'B-PERS', 'I-PERS', 'B-LOC', 'O']
sample_ner_labels_gt = ['O', 'B-PERS', 'I-PERS', 'B-LOC', 'I-LOC']

sample_seqeval_output = seqeval_evaluation([sample_ner_labels_pred],
                                           [sample_ner_labels_gt])
