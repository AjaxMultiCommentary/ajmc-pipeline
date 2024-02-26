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
    "printed_marginalia",
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

        self.ajmc_id = ajmc_id
        self.commentary = cc.CanonicalCommentary.from_json(json_path=json_path)
        self.filename = f"{ajmc_id}.xml"

    def facsimile(self, page):
        return f"{self.ajmc_id}/{page.id}/full/max/0/default.png"

    def to_tei(self):
        sections = []
        for section in self.commentary.children.sections:
            pages = []

            for page in section.children.pages:
                pages.append(E.pb(n=page.id, facs=self.facsimile(page)))

                for region in page.children.regions:
                    if region.region_type in TEI_REGION_TYPES:
                        pages.append(
                            E.p(
                                region.text,
                                type=region.region_type,
                                n="-".join(
                                    [
                                        str(region.word_range[0]),
                                        str(region.word_range[1]),
                                    ]
                                ),
                            )
                        )

            section_el = E.div(
                E.head(section.section_title),
                *pages,
                type="textpart",
                subtype="section",
                n=" ".join(section.section_types),
            )
            section_el.attrib["{http://www.w3.org/XML/1998/namespace}id"] = section.id
            sections.append(section_el)

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
                    E.div(
                        E.title(commentary_data["title"]),
                        *sections,
                        type="textpart",
                        subtype="commentary",
                    )
                )
            ),
        )

    def export(self):
        tei = self.to_tei()

        with open(self.filename, "wb") as f:
            lxml_etree.indent(tei, space="\t")
            f.write(lxml_etree.tostring(tei, encoding="utf-8", xml_declaration=True))  # type: ignore
