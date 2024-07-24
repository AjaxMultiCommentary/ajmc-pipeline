import re
import requests
import unicodedata

from lxml import etree

NAMESPACES = {
    "tei": "http://www.tei-c.org/ns/1.0",
    "xml": "http://www.w3.org/XML/1998/namespace",
}

SELECTOR_REGEX = re.compile(r"n=(?P<n>\d+)\[(?P<offset>\d+)\]")


class Chunk:
    def __init__(self, text: str, start_offset: int, elem):
        self.text = text
        self.start_offset = start_offset
        self.elem = elem


class TEI2TextMapper:
    def __init__(self, text_url: str, chunk_by: str = "tei:l"):
        parser = etree.XMLParser(remove_blank_text=True)
        req = requests.get(text_url)

        self.tree = etree.XML(req.text, parser)
        self.chunk_by = chunk_by

        all_text = self.tree.xpath(f"//{self.chunk_by}/text()", namespaces=NAMESPACES)[
            0
        ].strip()

        self.text = unicodedata.normalize("NFC", all_text)
        self._chunks = []

        str_offset = 0

        for chunk in self.tree.iterfind(f".//{self.chunk_by}", namespaces=NAMESPACES):
            chunk_text = chunk.text

            if chunk_text is not None:
                self._chunks.append(
                    Chunk(
                        text=chunk_text,
                        start_offset=str_offset,
                        elem=chunk,
                    )
                )
                str_offset += len(chunk_text)

    def selector_to_offsets(self, selector: str):
        [f, e] = selector.split(":")

        f_matches = SELECTOR_REGEX.search(f)
        e_matches = SELECTOR_REGEX.search(e)

        if f_matches is not None and e_matches is not None:
            f_n = f_matches.group("n")
            e_n = e_matches.group("n")
            f_offset = f_matches.group("offset")
            e_offset = e_matches.group("offset")

            f_chunk = [chunk for chunk in self._chunks if chunk.elem.get("n") == f_n][0]
            e_chunk = [chunk for chunk in self._chunks if chunk.elem.get("n") == e_n][0]

            return [
                f_chunk.start_offset + int(f_offset),
                e_chunk.start_offset + int(e_offset),
            ]

    def offsets_to_selector(self, offsets: list[int]):
        first_chunk = [
            chunk
            for chunk in self._chunks
            if offsets[0] >= chunk.start_offset
            and offsets[0] < chunk.start_offset + len(chunk.text)
        ][0]
        last_chunk = [
            chunk
            for chunk in self._chunks
            if offsets[1] > chunk.start_offset
            and chunk.start_offset + len(chunk.text) >= offsets[1]
        ][0]

        return f"""{self.chunk_by.replace(':', '-')}@n={
            first_chunk.elem.get('n')}[{
            offsets[0] - first_chunk.start_offset}]:{
            self.chunk_by.replace(':', '-')}@n={
            last_chunk.elem.get('n')}[{
            offsets[1] - last_chunk.start_offset}]"""
