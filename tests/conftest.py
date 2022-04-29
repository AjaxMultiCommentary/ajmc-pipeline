import pytest


@pytest.fixture(scope="session")
def sample_tsv_path():
    # return "/Users/matteo/Documents/AjaxMultiCommentary/HIPE2022-corpus/data/release/v2.0/HIPE-2022-v2.0-ajmc-dev-en.tsv"
    return '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/epibau/EpibauCorpus/data/release/v0.3/EpiBau-data-v0.3-test.tsv'
