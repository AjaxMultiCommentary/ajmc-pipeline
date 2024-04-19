import json

import lunr
from lazy_objects.lazy_objects import lazy_property

import ajmc.commons.unicode_utils as unicode_utils
import ajmc.commons.variables as variables
import ajmc.text_processing.canonical_classes as canonical_classes

DEFAULT_SEARCH_INDEX_EXPORT_LOCATION = "."


class CommentaryIndex:
    def __init__(self, commentary_id: str) -> None:
        self.commentary_id = commentary_id
        self.commentary_path = variables.get_comm_canonical_path_from_ocr_run_pattern(
            commentary_id, variables.COMM_BEST_OCR_GLOB
        )

    @lazy_property
    def canonical_commentary(self):
        return canonical_classes.CanonicalCommentary.from_json(self.commentary_path)

    @lazy_property
    def documents(self):
        return [
            {
                "id": page.id,
                "raw_text": unicode_utils.remove_diacritics(page.text),
                "text": page.text,
                "regions": [region.to_json() for region in page.children.regions],
                "words": [(word.index, word.to_json()) for word in page.children.words],
            }
            for page in self.canonical_commentary.children.pages
        ]


class SearchIndex:
    def __init__(self, commentary_ids: list[str]) -> None:
        self.commentary_ids = commentary_ids

    @lazy_property
    def documents(self):
        docs = []

        for commentary_id in self.commentary_ids:
            commentary_index = CommentaryIndex(commentary_id)
            docs = docs + commentary_index.documents

        return docs
    
    def export(self, to=DEFAULT_SEARCH_INDEX_EXPORT_LOCATION):
        serialized_index = self.search_index.serialize()

        with open(f"{to}/search_index.json", "w") as f:
            json.dump(serialized_index, f)

    @lazy_property
    def search_index(self):
        return lunr.lunr(ref="id", fields=("raw_text",), documents=self.documents)

    def search(self, search_string: str) -> list[dict]:
        cleaned_string = unicode_utils.remove_diacritics(search_string)

        return self.search_index.search(cleaned_string)
    
class PublicDomainSearchIndex(SearchIndex):
    def __init__(self) -> None:
        super().__init__(variables.PD_COMM_IDS)

class CopyrightSearchIndex(SearchIndex):
    def __init__(self) -> None:
        super().__init__(variables.COPYRIGHT_COMM_IDS)

class AjaxSearchIndex(SearchIndex):
    def __init__(self) -> None:
        super().__init__(variables.PD_COMM_IDS + variables.COPYRIGHT_COMM_IDS)

def test_search_index():
    index = SearchIndex(['sophoclesplaysa05campgoog'])

    results = index.search("Ajax")

    assert len(results) > 0

    grc_results = index.search("Αἶας")

    assert len(grc_results) > 0
