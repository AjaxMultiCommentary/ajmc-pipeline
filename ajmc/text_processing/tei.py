import ajmc.text_processing.canonical_classes as cc
import ajmc.text_processing.exportable_commentary as export
import ajmc.text_processing.zotero as zotero
import ajmc.commons.variables as variables
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


class TEIDocument(export.ExportableCommentary):
    def __init__(self, ajmc_id, bibliographic_data) -> None:
        super().__init__(ajmc_id, bibliographic_data)

        self.tei = None

    def authors(self):
        return [
            E.author(f"{a['firstName']} {a['lastName']}")
            for a in self.bibliographic_data["creators"]
        ]

    def frontmatter(self):
        return E.teiHeader(
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
                    section_heading = self.section_head(region)

                    if section_heading is not None:
                        page_el.append(section_heading)

                    if region.region_type == "footnote":
                        page_el.append(
                            E.note(
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
                        region_el = E.p(type=region.region_type)
                        for line in region.children.lines:
                            for w in self.words(line.word_range):
                                region_el.append(w)

                            region_el.append(E.lb())
                        page_el.append(region_el)

        return page_el

    def section_head(self, region):
        region_heading = TEI_REGION_LABELS.get(region.region_type)

        if region_heading != "":
            return E.head(region_heading)

        return None

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
            self.frontmatter(),
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
                isinstance(entity, export.PrimaryFullEntity)
                or (
                    entity.label in export.EXPORTED_ENTITY_LABELS  # type: ignore
                    and entity.wikidata_id is not None  # type: ignore
                )
            ):
                if entity == current_entity and current_el is not None:
                    current_el.append(E.w(word.text))
                else:
                    current_entity = entity

                    if current_el is not None:
                        words.append(current_el)

                    if isinstance(entity, export.PrimaryFullEntity):
                        current_el = E.ref(E.w(word.text), target=entity.url)
                    else:
                        current_el = E.ref(E.w(word.text), target=entity.wikidata_id)  # type: ignore
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
        zotero_data = zotero.get_zotero_data(commentary["zotero_id"])
        doc = TEIDocument(commentary["pid"], zotero_data)

        doc.export()
