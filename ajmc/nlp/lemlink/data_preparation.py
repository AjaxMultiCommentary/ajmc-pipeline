import re
import unicodedata

import requests
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

        all_text = ''.join(self.tree.xpath(f"//{self.chunk_by}/text()", namespaces=NAMESPACES)).strip()

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


#%%
from pathlib import Path
from ajmc.nlp.token_classification.data_preparation.hipe_iob import read_lemlink_tsv

sample_tsv_path = Path('/scratch/sven/ajmc_data/lemma-linkage-corpus/data/release/v1.0.beta/lemlink-v1.0.beta-test_NOCOMMENT.tsv')

data = read_lemlink_tsv(sample_tsv_path)
data = data.to_dict(orient='list')

mapper = TEI2TextMapper('http://raw.githubusercontent.com/gregorycrane/Wolf1807/master/ajax-2019/ajax-lj.xml')

for i in range(len(data['ANCHOR_TARGET'])):
    if data['ANCHOR_TARGET'][i] != '_':
        sample_selector = data['ANCHOR_TARGET'][i]
        sample_text = data['ANCHOR_TEXT'][i]
        break
        # text = mapper.text[mapper.selector_to_offsets(data['ANCHOR_TARGET'][i])[0]:mapper.selector_to_offsets(data['ANCHOR_TARGET'][i])[1]]
        # if text != data['ANCHOR_TEXT'][i]:
        #     print(i, text, ' |||| ', data['ANCHOR_TEXT'][i])
        # else:
        #     print(i, 'OK')
        # break

offsets = mapper.selector_to_offsets(sample_selector)
sample_text
mapper.text
