from ajmc.nlp.token_classification.data_preparation.utils import align_labels_to_tokenized, align_to_tokenized
from tests import sample_objects as so

def test_align_labels_to_tokenized():

        null_label = -100

        offsets = [None]
        true_aligned = [null_label]

        for i, _ in enumerate(so.sample_ner_labels_gt):
            offsets.append(i)
            offsets.append(i)
            true_aligned.append(so.sample_labels_to_ids[so.sample_ner_labels_gt[i]])
            true_aligned.append(null_label)


        aligned = align_labels_to_tokenized(offsets, so.sample_ner_labels_gt, so.sample_labels_to_ids,
                                            label_all_tokens=False,
                                            null_label=null_label)

        assert aligned == true_aligned



def test_align_to_tokenized():

    offsets = []
    true_aligned = []
    for i, _ in enumerate(so.sample_ner_labels_gt):
        offsets.append(i)
        offsets.append(i)
        true_aligned.append(so.sample_ner_labels_gt[i])
        true_aligned.append(None)

    aligned = align_to_tokenized(offsets, so.sample_ner_labels_gt)
    assert aligned == true_aligned

