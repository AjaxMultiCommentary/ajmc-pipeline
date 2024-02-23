import ajmc.text_processing.canonical_classes as cc
import ajmc.commons.variables as vars
import lxml.builder as lxml_builder
import lxml.etree as lxml_etree
import os


E = lxml_builder.ElementMaker(
    namespace="http://www.tei-c.org/ns/1.0",
    nsmap={None: "http://www.tei-c.org/ns/1.0"},
)

TEI_REGION_TYPES = [
    "app_crit",
    "appendix",
    "bibliography",
    "commentary",
    "footnote",
    "index",
    "introduction",
    "preface",
    "primary_text",
    "table_of_contents",
    "title",
    "translation",
]

COMMENTARIES_DATA = {
    "sophoclesplaysa05campgoog": {
        "title": "Sophocles: The Plays and Fragments: Volume 7: The Ajax",
        "author": "R. C. Jebb",
    }
}


class TEIDocument:
    def __init__(self, ajmc_id) -> None:
        canonical_path = vars.COMMS_DATA_DIR / ajmc_id / "canonical"
        filename = [
            f for f in os.listdir(canonical_path) if f.endswith("_tess_retrained.json")
        ][0]
        json_path = canonical_path / filename

        self.commentary = cc.CanonicalCommentary.from_json(json_path=json_path)
        self.filename = f"{ajmc_id}.xml"

    def to_tei(self):
        sections = []
        for section in self.commentary.children.sections:
            pages = []

            for page in section.children.pages:
                regions = []

                for region in page.children.regions:
                    if region.region_type in TEI_REGION_TYPES:
                        regions.append(E.div(region.text, type=region.region_type))
                pages.append(E.div(*regions, type="page", n=page.id))

            sections.append(
                E.div(*pages, type=" ".join(section.section_types), n=section.id)
            )

        commentary_data = COMMENTARIES_DATA[self.commentary.id]  # type: ignore

        return E.TEI(
            E.teiHeader(
                E.fileDesc(
                    E.titleStmt(
                        E.title(commentary_data["title"]),
                        E.author(commentary_data["author"]),
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
                    E.title(commentary_data["title"]),
                    *sections,
                )
            ),
        )

    def export(self):
        tei = self.to_tei()

        with open(self.filename, "wb") as f:
            lxml_etree.indent(tei, space="\t")
            f.write(lxml_etree.tostring(tei, encoding="utf-8", xml_declaration=True))  # type: ignore
