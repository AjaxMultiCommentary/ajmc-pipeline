import ajmc.text_processing.tei as tei
import pytest
import random

from lxml.etree import _Element


# FIXME: sample_objects.py is failing, so I'm just inlining the
# dummy objects here for now.
sample_commentary_id = "cu31924087948174"
sample_zotero_data = {
    "key": "3PSZEAK5",
    "version": 2230,
    "itemType": "book",
    "title": "Sophocles",
    "creators": [
        {"creatorType": "author", "firstName": "Lewis", "lastName": "Campbell"}
    ],
    "abstractNote": "",
    "series": "",
    "seriesNumber": "",
    "volume": "2",
    "numberOfVolumes": "",
    "edition": "",
    "place": "Oxford",
    "publisher": "Clarendon Press",
    "date": "1881",
    "numPages": "612",
    "language": "eng, grc",
    "ISBN": "",
    "shortTitle": "",
    "url": "http://archive.org/details/cu31924087948174",
    "accessDate": "2020-12-30T15:26:13Z",
    "archive": "",
    "archiveLocation": "",
    "libraryCatalog": "Internet Archive",
    "callNumber": "",
    "rights": "",
    "extra": "Citation Key: campbell_sophocles_1881\nQID: Q123679674\nPublic Domain Year: 1978\nURN: tlg0011.tlg003.ajmc-cam",
    "tags": [],
    "collections": ["NTFEUW62"],
    "relations": {},
    "dateAdded": "2020-12-30T15:26:13Z",
    "dateModified": "2024-01-23T11:45:43Z",
}


@pytest.fixture(scope="module")
def document():
    return tei.TEIDocument(sample_commentary_id, sample_zotero_data)


class TestTEIDocument:
    def test_authors(self, document):
        author = random.choice(document.authors())

        assert isinstance(author, _Element)

    def test_page_transcription(self, document):
        page = random.choice(document.commentary.children.pages)

        assert isinstance(document.page_transcription(page), _Element)

    def test_title(self, document):
        assert document.title() == sample_zotero_data["title"]

    def test_to_tei(self, document):
        assert document.to_tei() is not None

    def test_facsimile(self, document):
        page = random.choice(document.commentary.children.pages)

        assert document.facsimile(page) == f"{document.ajmc_id}/{page.id}"
