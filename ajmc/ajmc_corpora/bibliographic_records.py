"""⚙️ WIP code process bibliographic records"""

from pathlib import Path
from typing import Union, List

from bs4 import BeautifulSoup


# from lazy_objects.lazy_objects import lazy_property, lazy_init


class DublinCoreRecord:

    #@lazy_init
    def __init__(self, soup: BeautifulSoup):
        pass

    def get_property_tag_text(self, tag_name: str) -> str:
        return self.soup.find(tag_name).text.lower()

    #@lazy_property
    def title(self) -> str:
        return self.get_property_tag_text('dcterms:title')

    #@lazy_property
    def creator(self) -> str:
        return self.get_property_tag_text('dcterms:creator')

    #@lazy_property
    def publisher(self) -> str:
        return self.get_property_tag_text('dcterms:publisher')

    #@lazy_property
    def language(self) -> str:
        return self.get_property_tag_text('dcterms:language')

    #@lazy_property
    def keywords(self) -> List[str]:
        return [s.text.lower() for s in self.soup.find_all('dcterms:subject')]

    #@lazy_property
    def keywords_string(self) -> str:
        return ' '.join(self.keywords)

    #@lazy_property
    def description(self) -> str:
        return self.get_property_tag_text('dcterms:description')

    #@lazy_property
    def whole_text(self) -> str:
        return ' '.join(
                [self.title, self.creator, self.publisher, self.language, self.keywords_string, self.description])


def get_records_list(xmls_dir: Union[Path, str]) -> List[BeautifulSoup]:
    records = []
    for path in Path(xmls_dir).glob('*.xml'):
        soup = BeautifulSoup(path.read_text(), features='xml')
        for record in soup.find_all('record'):
            records.append(record)

    return records
