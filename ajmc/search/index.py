import ajmc.text_processing.canonical_classes as canonical_classes
import ajmc.commons.variables as variables
import lunr

from lazy_objects.lazy_objects import lazy_property

class Index:
    def __init__(self, commentary_id: str) -> None:
        self.commentary_id = commentary_id
        self.commentary_path = variables.get_comm_canonical_path_from_pattern(
            commentary_id, variables.COMM_BEST_OCR_GLOB
        )
        self.canonical_commentary = canonical_classes.CanonicalCommentary.from_json(
            self.commentary_path
        )

    @lazy_property
    def search_index(self):
        documents = [
            {"id": page.id, "text": "\n".join([l.text for l in page.children.lines])}
            for page in self.canonical_commentary.children.pages
        ]
        return lunr.lunr(ref="id", fields=("text",), documents=documents)


def test_search_index():
    index = Index("sophoclesplaysa05campgoog")

    assert type(index.canonical_commentary) == canonical_classes.CanonicalCommentary
    assert len(index.search_index.search('Ajax')) > 0
