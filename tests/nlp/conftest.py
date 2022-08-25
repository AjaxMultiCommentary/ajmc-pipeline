import pytest


# Todo 👁️ see that you unify this with `sample_objects.py`
labels = ['O', 'B-PERS', 'I-PERS', 'B-LOC', 'I-LOC']


@pytest.fixture(scope="session")
def test_labels():
    return labels


@pytest.fixture(scope="session")
def test_labels_to_ids():
    return {l: i for i, l in enumerate(labels)}


@pytest.fixture(scope="session")
def test_ids_to_labels():
    return {i: l for i, l in enumerate(labels)}

@pytest.fixture(scope="session")
def test_tokens_to_words_offsets():
    offsets = []
    for i, _ in enumerate(labels):
        offsets.append(i)
        offsets.append(i)
    return offsets