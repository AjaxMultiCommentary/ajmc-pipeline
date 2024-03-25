import ajmc.text_processing.canonical_classes as cc
import ajmc.text_processing.exportable_commentary as export
import regex

# note that the `U` and `UNICDOE` flags are redundant
# https://docs.python.org/3/library/re.html#flags
REFERENCE_REGEX = regex.compile(r"(\w+)(?:\[(\d+)\])?", regex.IGNORECASE)


class CTS_URN:
    """
    Some methods in this class were borrowed
    from/inspired by
    https://github.com/Capitains/MyCapytain/blob/dev/MyCapytain/common/reference/_capitains_cts.py
    """

    prefix: str
    protocol: str
    namespace: str
    text_group: str | None
    work: str | None
    version: str | None
    exemplar: str | None
    references: str | None
    citations: list[str] | None
    subsections: list[str] | None
    indexes: list[str] | None

    def __init__(self, urn: str) -> None:
        self.__urn = None

        self.__parse__(urn)

    def __parse__(self, urn_s: str):
        self.__urn = urn_s.split("#")[0]

        urn = self.__urn.split(":")

        if isinstance(urn, list) and len(urn) > 2:
            self.prefix = urn[0]
            self.protocol = urn[1]
            self.namespace = urn[2]

            if len(urn) == 5:
                self.references = urn[4]
        else:
            raise ValueError(f"Invalid URN {urn_s}")

    def parse(self, s: str):
        return s.split(":")

    def passage_component(self) -> str:
        if self.citations is None:
            return ""

        if self.subsections is None:
            return f"{'-'.join(self.citations)}"

        if self.indexes is None:
            return f"{self.citations[0]}@{self.subsections[0]}-{self.citations[1]}@{self.subsections[1]}"

        return f"{self.citations[0]}@{self.subsections[0]}[{self.indexes[0]}]-{self.citations[1]}@{self.subsections[1]}[{self.indexes[1]}]"

    def work_component(self) -> str:
        if self.text_group is None:
            return ""

        if self.work is None:
            return f"{self.text_group}"

        if self.version is None:
            return f"{self.text_group}.{self.work}"

        if self.exemplar is None:
            return f"{self.text_group}.{self.work}.{self.version}"

        return f"{self.text_group}.{self.work}.{self.version}.{self.exemplar}"


class Glossa:
    attributes: dict
    content: str
    lemma: str
    lemma_start_offset: int
    lemma_end_offset: int
    urn: CTS_URN

    def __init__(self, canonical_lemma: cc.CanonicalLemma) -> None:
        self.__lemma = canonical_lemma

    def get_content(self):
        lemma_words = self.__lemma.children.words


class MarkdownCommentary(export.ExportableCommentary):
    def __init__(self, ajmc_id, bibliographic_data) -> None:
        self.markdown = None

        super().__init__(ajmc_id, bibliographic_data)

    def frontmatter(self):
        creators = [
            dict(
                first_name=a["firstName"],
                last_name=a["lastName"],
                creator_type=a["creatorType"],
            )
            for a in self.bibliographic_data["creators"]
        ]
        edition = self.bibliographic_data["edition"]
        languages = self.bibliographic_data["language"].split(", ")
        place = self.bibliographic_data["place"]
        publication_date = self.bibliographic_data["date"]
        public_domain_year = self.bibliographic_data["extra"]["Public Domain Year"]
        publisher = self.bibliographic_data["publisher"]
        source_url = self.bibliographic_data["url"]
        title = self.bibliographic_data["title"]
        wikidata_qid = self.bibliographic_data["extra"]["QID"]
        urn = self.bibliographic_data["extra"]["URN"]
        zotero_id = self.bibliographic_data["key"]
        zotero_link = (
            self.bibliographic_data.get("links", {}).get("alternate", {}).get("href")
        )

        return dict(
            creators=creators,
            edition=edition,
            languages=languages,
            metadata=self.commentary.metadata,  # type: ignore
            pid=self.commentary.id,  # type: ignore
            place=place,
            publication_date=publication_date,
            public_domain_year=public_domain_year,
            publisher=publisher,
            source_url=source_url,
            title=title,
            urn=f"urn:cts:greekLit:{urn}",
            wikidata_qid=wikidata_qid,
            zotero_id=zotero_id,
            zotero_link=zotero_link,
        )

    def glosses(self):
        for i, lemma in enumerate(self.commentary.children.lemmas):
            try:
                next_lemma = self.commentary.children.lemmas[i + 1]
                glossa_words = self.get_words_between_lemmas(lemma, next_lemma)
            except IndexError:
                pass

    def get_words_between_lemmas(
        self, lemma: cc.CanonicalLemma, next_lemma: cc.CanonicalLemma
    ):
        [_lemma_start, lemma_end] = lemma.word_range
        [next_lemma_start, _next_lemma_end] = next_lemma.word_range

        return self.commentary.children.words[lemma_end + 1 : next_lemma_start]
