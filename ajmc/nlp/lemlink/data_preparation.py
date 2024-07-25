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

        self._chunks = []

        str_offset = 0

        for chunk in self.tree.iterfind(f".//{self.chunk_by}", namespaces=NAMESPACES):
            chunk_text = chunk.text

            if chunk_text is not None:
                t = unicodedata.normalize('NFC', chunk_text)

                self._chunks.append(
                    Chunk(
                        text=t,
                        start_offset=str_offset,
                        elem=chunk,
                    )
                )
                str_offset += len(t)

        self.text = ''.join(c.text for c in self._chunks)

    def selector_to_offsets(self, selector: str):
        [f, l] = selector.split(":")

        f_matches = SELECTOR_REGEX.search(f)
        l_matches = SELECTOR_REGEX.search(l)

        if f_matches is not None and l_matches is not None:
            f_n = f_matches.group("n")
            l_n = l_matches.group("n")
            f_offset = f_matches.group("offset")
            l_offset = l_matches.group("offset")

            f_chunk = [chunk for chunk in self._chunks if chunk.elem.get("n") == f_n][0]
            l_chunk = [chunk for chunk in self._chunks if chunk.elem.get("n") == l_n][0]

            return [
                f_chunk.start_offset + int(f_offset),
                l_chunk.start_offset + int(l_offset),
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
