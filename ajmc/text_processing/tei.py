import ajmc.text_processing.canonical_classes as cc
import ajmc.commons.variables as variables
import json
import lxml.builder as lxml_builder
import lxml.etree as lxml_etree
import requests
import os

from typing import Tuple

"""
This module --- which is really more of a script --- enables exporting
public domain commentaries to their TEI representation. You can run it
from the command line with `python ajmc/text_processing/tei.py`, or you
can import it (`import ajmc.text_processing.tei as tei`) and use it in
your own script.

It expects a few environment variables to be set:

- `AJMC_DATA_DIR`: As with the bulk of this `ajmc` library, you'll need to point to a local copy of the AjMC canonical commentaries data directory.
- `AJMC_API_URL`: This module uses the AjMC API to get information about the commentaries. It's most likely easiest to point to a local version.
- `ZOTERO_API_URL`: The URL for the AjMC Zotero group. The example URL uses the string `GROUP` to indicate that you should subsitute in the group ID.
- `ZOTERO_API_TOKEN`: Your Zotero API token for accessing the Zotero group.
"""

E = lxml_builder.ElementMaker(
    namespace="http://www.tei-c.org/ns/1.0",
    nsmap={None: "http://www.tei-c.org/ns/1.0"},
)

EXPORTED_ENTITY_LABELS = ["work.primlit", "pers.author"]

TEI_REGION_LABELS = {
    "app_crit": "Critical Apparatus",
    "appendix": "Appendix",
    "bibliography": "Bibliography",
    "commentary": "Commentary",
    "footnote": "Footnote",
    "index": "Index",
    "introduction": "Introduction",
    "line": "",
    "line_region": "",
    "preface": "Preface",
    "primary_text": "",
    "printed_marginalia": "",
    "table_of_contents": "Table of Contents",
    "title": "Title",
    "translation": "Translation",
}
TEI_REGION_TYPES = TEI_REGION_LABELS.keys()

WIKIDATA_HUCIT_MAPPINGS = {}

with open(os.path.dirname(__file__) + "/wikidata_hucit_mappings.json") as f:
    WIKIDATA_HUCIT_MAPPINGS = json.load(f)


def contains_primary_full(entities):
    return any([is_primary_full(entity) for entity in entities])


def is_primary_full(entity):
    return entity.label == "primary-full"


def is_entity_in_range(entity, word_range):
    return (
        word_range[0] <= entity.word_range[0]
        and (word_range[1] + 1) >= entity.word_range[1]
    )


def remove_trailing_character(s: str, c: str) -> str:
    if s.endswith(c):
        return s[0:-1]

    return s


def transform_f(s: str) -> str:
    if s.endswith("f."):
        without_f = s.replace("f.", "")

        try:
            n = int(without_f)
            return f"{n}-{n+1}"
        except ValueError:
            # Return the scope without "f." because we're using it in a CTS URN
            return without_f

    return s


class PrimaryFullEntity:
    def __init__(
        self,
        cts_urn: str,
        scopes: list[cc.CanonicalEntity],
        words: list[cc.CanonicalWord],
    ):
        self.cts_urn = cts_urn
        self.scopes = scopes
        self.words = words
        self.word_range = [words[0].word_range[0], words[-1].word_range[1]]
        self.url = self.to_url()

    def to_url(self):
        if len(self.scopes) == 0:
            return f"https://scaife.perseus.org/reader/{self.cts_urn}"
        else:
            return f"https://scaife.perseus.org/reader/{self.cts_urn}{self.resolve_scopes()}"

    def resolve_scopes(self):
        scope_first = self.scopes[0]
        scope_last = self.scopes[-1]
        scope_words = [
            w
            for w in self.words
            if w.word_range[0]
            in range(scope_first.word_range[0], scope_last.word_range[1] + 1)
        ]
        s = (
            "".join([w.text for w in scope_words])
            .replace("(", "")
            .replace(")", "")
            .replace(";", "")
            .replace(":", "")
            .replace("ff.", "")
        )
        s = transform_f(s)
        s = remove_trailing_character(s, ",")
        s = remove_trailing_character(s, ".")

        if len(s) > 0:
            return f":{s}"

        return ""


def make_primary_full_entities(commentary: cc.CanonicalCommentary):
    all_entities = commentary.children.entities
    primary_fulls = []

    for entity in all_entities:
        if entity.label == "primary-full":
            related_entities = [
                e for e in all_entities if is_entity_in_range(e, entity.word_range)
            ]
            primlits = [e for e in related_entities if e.label == "work.primlit"]
            scopes = [e for e in related_entities if e.label == "scope"]

            wikidata_id = next((e.wikidata_id for e in primlits), None)

            if wikidata_id is None:
                wikidata_id = next((e.wikidata_id for e in scopes), None)

                if wikidata_id is None:
                    continue

            cts_urn = WIKIDATA_HUCIT_MAPPINGS.get(wikidata_id, {}).get("cts_urn")

            if cts_urn is None or cts_urn == "":
                continue

            entity_words = commentary.children.words[
                entity.word_range[0] : (entity.word_range[1] + 1)
            ]

            if len(entity_words) == 0:
                continue

            primary_fulls.append(PrimaryFullEntity(cts_urn, scopes, entity_words))

    return primary_fulls


def get_zotero_data(zotero_id):
    resp = requests.get(
        f'{os.getenv("ZOTERO_API_URL", "https://api.zotero.org/groups/GROUP")}/items/{zotero_id}',
        headers={"Authorization": f"Bearer {os.getenv('ZOTERO_API_TOKEN', '')}"},
    )
    return resp.json()["data"]


class TEIDocument:
    def __init__(self, ajmc_id, bibliographic_data) -> None:
        canonical_path = variables.COMMS_DATA_DIR / ajmc_id / "canonical"
        filename = [
            f for f in os.listdir(canonical_path) if f.endswith("_tess_retrained.json")
        ][0]
        json_path = canonical_path / filename

        self.ajmc_id = ajmc_id
        self.bibliographic_data = bibliographic_data
        self.commentary = cc.CanonicalCommentary.from_json(json_path=json_path)
        self.filename = f"tei/{ajmc_id}.xml"
        self.primary_full_entities = make_primary_full_entities(self.commentary)
        self.tei = None

    def authors(self):
        return [
            E.author(f"{a['firstName']} {a['lastName']}")
            for a in self.bibliographic_data["creators"]
        ]

    def facsimile(self, page):
        return f"{self.ajmc_id}/{page.id}"

    def get_entity_for_word(self, word):
        primary_full_entity = next(
            (
                e
                for e in self.primary_full_entities
                if word.index in range(e.word_range[0], e.word_range[1] + 1)
            ),
            None,
        )

        if primary_full_entity is not None:
            return primary_full_entity

        return next(
            (
                entity
                for entity in self.commentary.children.entities
                if word.index in range(entity.word_range[0], entity.word_range[1])
            ),
            None,
        )

    def page_transcription(self, page):
        page_el = E.div(E.pb(n=page.id, facs=self.facsimile(page)))

        # If there are no regions with text on the page, fall back to
        # the page's text
        if "".join([r.text for r in page.children.regions]).strip() == "":
            page_el.append(
                E.p(
                    *self.words(page.word_range),
                    type="page",
                    n="-".join(
                        [
                            str(page.word_range[0]),
                            str(page.word_range[1]),
                        ]
                    ),
                )
            )
        else:
            for region in page.children.regions:
                if region.region_type in TEI_REGION_TYPES:
                    if region.region_type == "footnote":
                        page_el.append(
                            E.note(
                                self.section_head(region),
                                *self.words(region.word_range),
                                place="foot",
                                n="-".join(
                                    [
                                        str(region.word_range[0]),
                                        str(region.word_range[1]),
                                    ]
                                ),
                            )
                        )
                    else:
                        page_el.append(
                            E.p(
                                self.section_head(region),
                                *self.words(region.word_range),
                                type=region.region_type,
                                n="-".join(
                                    [
                                        str(region.word_range[0]),
                                        str(region.word_range[1]),
                                    ]
                                ),
                            )
                        )

        return page_el

    def section_head(self, region):
        region_heading = TEI_REGION_LABELS.get(region.region_type)

        if region_heading != "":
            return E.head(region_heading)

        return ""

    def title(self):
        return self.bibliographic_data["title"]

    def to_tei(self):
        if self.tei is not None:
            return self.tei

        sections = []
        for section in self.commentary.children.sections:
            pages = [self.page_transcription(page) for page in section.children.pages]

            section_el = E.div(
                E.head(section.section_title),
                *pages,
                type="textpart",
                subtype="section",
                n=" ".join(section.section_types),
            )
            section_el.attrib["{http://www.w3.org/XML/1998/namespace}id"] = section.id
            sections.append(section_el)

        self.tei = E.TEI(
            E.teiHeader(
                E.fileDesc(
                    E.titleStmt(
                        E.title(self.title()),
                        *self.authors(),
                    ),
                    E.publicationStmt(
                        E.publisher("Ajax Multi-Commentary"),
                        E.availability(status="free"),
                    ),
                    E.sourceDesc(
                        E.p(
                            "Created from public domain scans of the public domain commentary"
                        )
                    ),
                ),
                E.revisionDesc(
                    E.change("Initial TEI export", when="2024-02-23", who="#AjMC")
                ),
            ),
            E.text(
                E.body(
                    E.div(
                        E.title(self.title()),
                        *sections,
                        type="textpart",
                        subtype="commentary",
                    )
                )
            ),
        )

        return self.tei

    """
    Iterate through the words in `word_range`, checking each word
    to see if it belongs to a primary full entity.

    If it does, create or update `current_entity` to include the
    `<w>` element for the word.

    If a new entity is encountered, push the current `current_entity` onto
    the `words` list and start a new `current_entity`.
    """

    def words(self, word_range: Tuple[int, int]):
        words = []
        current_entity = None
        current_el = None

        for word in self.commentary.children.words[word_range[0] : word_range[1] + 1]:
            entity = self.get_entity_for_word(word)

            if entity is not None and (
                isinstance(entity, PrimaryFullEntity)
                or (
                    entity.label in EXPORTED_ENTITY_LABELS
                    and entity.wikidata_id is not None
                )
            ):
                if entity == current_entity and current_el is not None:
                    current_el.append(E.w(word.text))
                else:
                    current_entity = entity

                    if current_el is not None:
                        words.append(current_el)

                    if isinstance(entity, PrimaryFullEntity):
                        current_el = E.ref(E.w(word.text), target=entity.url)
                    else:
                        current_el = E.ref(E.w(word.text), target=entity.wikidata_id)
            else:
                words.append(E.w(word.text))

        return words

    def export(self):
        tei = self.to_tei()

        with open(self.filename, "wb") as f:
            lxml_etree.indent(tei, space="\t")
            f.write(lxml_etree.tostring(tei, encoding="utf-8", xml_declaration=True))  # type: ignore


if __name__ == "__main__":
    commentaries = requests.get(
        f"{os.getenv('AJMC_API_URL', 'https://ajmc.unil.ch/api')}/commentaries?public=true"
    )

    for commentary in commentaries.json()["data"]:
        zotero_data = get_zotero_data(commentary["zotero_id"])
        doc = TEIDocument(commentary["pid"], zotero_data)

        doc.export()
