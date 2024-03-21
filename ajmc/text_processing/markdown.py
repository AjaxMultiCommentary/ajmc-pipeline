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
        self.__parsed = self.__parse__(urn)

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
    start_offset: int
    end_offset: int
    urn: CTS_URN

    def __init__(self) -> None:
        pass


class MarkdownCommentary:
    def __init__(self) -> None:
        pass
