import ajmc.text_processing.canonical_classes as cc
import ajmc.commons.variables as vars
import lxml.builder as lxml_builder
import lxml.etree as lxml_etree
import requests
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


def get_zotero_data(zotero_id):
    resp = requests.get(
        f'{os.getenv("ZOTERO_API_URL", "https://api.zotero.org/groups/GROUP/")}/items/{zotero_id}',
        headers={"Authorization": f"Bearer {os.getenv('ZOTERO_API_TOKEN', '')}"},
    )
    return resp.json()["data"]


class TEIDocument:
    def __init__(self, ajmc_id, bibliographic_data) -> None:
        canonical_path = vars.COMMS_DATA_DIR / ajmc_id / "canonical"
        filename = [
            f for f in os.listdir(canonical_path) if f.endswith("_tess_retrained.json")
        ][0]
        json_path = canonical_path / filename

        self.ajmc_id = ajmc_id
        self.bibliographic_data = bibliographic_data
        self.commentary = cc.CanonicalCommentary.from_json(json_path=json_path)
        self.filename = f"tei/{ajmc_id}.xml"
        self.tei = None

    def authors(self):
        return [
            E.author(f"{a['firstName']} {a['lastName']}")
            for a in self.bibliographic_data["creators"]
        ]

    def facsimile(self, page):
        return f"{self.ajmc_id}/{page.id}"

    def title(self):
        return self.bibliographic_data["title"]

    def to_tei(self):
        if self.tei is not None:
            return self.tei

        sections = []
        for section in self.commentary.children.sections:
            pages = []

            for page in section.children.pages:
                page_el = E.div(E.pb(n=page.id, facs=self.facsimile(page)))

                for region in page.children.regions:
                    if region.region_type in TEI_REGION_TYPES:
                        page_el.append(
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

                pages.append(page_el)

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

    def export(self):
        tei = self.to_tei()

        with open(self.filename, "wb") as f:
            lxml_etree.indent(tei, space="\t")
            f.write(lxml_etree.tostring(tei, encoding="utf-8", xml_declaration=True))  # type: ignore


if __name__ == "__main__":
    commentaries = requests.get(
        f"{os.getenv('AJMC_API_URL', '')}/commentaries?public=true"
    )

    for commentary in commentaries.json()["data"]:
        zotero_data = get_zotero_data(commentary["zotero_id"])
        doc = TEIDocument(commentary["pid"], zotero_data)

        doc.export()
