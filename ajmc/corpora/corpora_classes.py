import json
import re
import typing
from abc import ABC, abstractmethod
from pathlib import Path

import beta_code
from bs4 import BeautifulSoup

from ajmc.commons.unicode_utils import is_charset_string
from ajmc.corpora import variables as vs
from ajmc.corpora.cleaning_utils import harmonise_linebreaks


class Corpus(ABC):
    """Mother class for all Corpus-objects. Can be used to instantiate child-object via ``auto_init``"""

    def __init__(self, corpus_id: str):
        self.id = corpus_id
        self.base_dir = vs.BASE_STORING_DIR / corpus_id
        self.metadata_path = self.base_dir / 'metadata.json'
        self.metadata = json.loads(self.metadata_path.read_text(encoding='utf-8'))


    @classmethod
    def auto_init(cls, corpus_id):
        """This classmethod can be used to create the right object from the mother class"""
        metadata_path = vs.BASE_STORING_DIR / corpus_id / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        return globals()[metadata['type']](corpus_id)

    @abstractmethod
    def data_dir(self) -> Path:
        pass

    @abstractmethod
    def files(self) -> typing.List[Path]:
        pass

    @abstractmethod
    def get_plain_text(self) -> str:
        pass

    def get_documents(self) -> typing.List[str]:
        return [d for d in self.get_plain_text().split('\n\n\n') if d.strip()]

    def get_regions(self) -> typing.List[str]:
        return [r for d in self.get_documents() for r in d.split('\n\n') if r.strip()]

    def get_lines(self) -> typing.List[str]:
        return [l for r in self.get_regions() for l in r.split('\n') if l.strip()]

    def get_words(self) -> typing.List[str]:
        return re.sub(r'\s+', ' ', self.get_plain_text()).split(' ')


    def get_chunks(self, chunk_size: int, unit: str = 'word') -> typing.List[str]:
        """Chunks the corpus into string ``chunk-size`` units.

        Args:
            chunk_size: The size of the chunks
            unit: The unit of the chunks. Can be 'word' or 'chararcter'
        """
        if unit == 'word':
            words = self.get_words()
            return [' '.join(words[i:i + chunk_size]) for i in range(0, len(self.get_words()), chunk_size)]
        elif unit == 'character':
            text = self.get_plain_text()
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        else:
            raise NotImplementedError(f'Unit {unit} not implemented ; should be either "word" or "character"')

    def write_plain_text(self, path: typing.Optional[typing.Union[str, Path]] = None):
        if path is None:
            (self.data_dir / 'corpus.txt').write_text(self.get_plain_text(), encoding='utf-8')
        else:
            Path(path).write_text(self.get_plain_text(), encoding='utf-8')


class PlainTextCorpus(Corpus):

    def __init__(self, corpus_id: str):
        super().__init__(corpus_id)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / 'data'

    @property
    def files(self) -> typing.List[Path]:
        return [self.data_dir / 'corpus_cleaned.txt']

    def get_plain_text(self) -> str:
        return self.files[0].read_text(encoding='utf-8')


class TeiCorpus(Corpus):

    def __init__(self, corpus_id: str):
        super().__init__(corpus_id)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / 'data'

    @property
    def files(self) -> typing.List[Path]:
        return list(self.data_dir.glob('*.xml'))

    def get_plain_text(self) -> str:
        if (self.base_dir / 'plaintext.txt').exists():
            return (self.base_dir / 'plaintext.txt').read_text(encoding='utf-8')

        full_text = ''
        for file in self.files:
            soup = BeautifulSoup(file.read_text(encoding='utf-8'), features='xml')
            for text in soup.find_all(re.compile(r'(?:tei:)?text')):
                full_text += text.text + '\n\n\n'

        (self.base_dir / 'plaintext.txt').write_text(full_text, encoding='utf-8')
        return full_text


class OGLCorpus(TeiCorpus):

    def __init__(self, corpus_id: str):
        super().__init__(corpus_id)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / self.id / 'data'

    @property
    def files(self) -> typing.List[Path]:
        return [p for p in self.data_dir.rglob('*.xml') if p.name[0] != '_']


class PerseusLegacyCorpus(Corpus):


    def __init__(self, corpus_id: str):
        super().__init__(corpus_id)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / 'data'

    @property
    def files(self) -> typing.List[Path]:
        return list(self.data_dir.rglob('*.xml'))


    def get_plain_text(self) -> str:
        if (self.base_dir / 'plaintext.txt').exists():
            return (self.base_dir / 'plaintext.txt').read_text(encoding='utf-8')

        text = '\n\n\n'.join([self.read_document(file) for file in self.files])
        text = harmonise_linebreaks(text)
        (self.base_dir / 'plaintext.txt').write_text(text, encoding='utf-8')
        return text


    @staticmethod
    def convert_soup_to_unicode(soup: BeautifulSoup) -> BeautifulSoup:
        greek_aliases = ['gr', 'gre', 'grc', 'greek']
        for element in soup.find_all():
            if element.attrs.get('xml:lang') in greek_aliases or element.attrs.get('lang') in greek_aliases:
                if not is_charset_string(element.text.replace(' ', ''), 'greek', threshold=0.9):
                    greek = beta_code.beta_code_to_greek(element.text).encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                    element.string = greek  # This screws up potential nested divs

        return soup


    @staticmethod
    def read_document(path: Path) -> str:
        soup = BeautifulSoup(path.read_text(encoding='utf-8'), features='xml')
        soup = PerseusLegacyCorpus.convert_soup_to_unicode(soup)
        return '\n\n\n'.join([text.text for text in soup.find_all('text')])


class LogeionCorpus(Corpus):

    def __init__(self, corpus_id: str = 'logeion'):
        super().__init__(corpus_id)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / 'data'

    @property
    def files(self) -> typing.List[Path]:
        return list((self.data_dir / 'cleaned').rglob('*.json'))


    def get_plain_text(self) -> str:
        if (self.base_dir / 'plaintext.txt').exists():
            return (self.base_dir / 'plaintext.txt').read_text(encoding='utf-8')
        text = ''
        for file in self.files:
            dict_ = json.loads(file.read_text(encoding='utf-8'))
            text += '\n'.join([v if v.endswith('.') else v + '.' for v in dict_.values()]) + '\n\n\n'

        (self.base_dir / 'plaintext.txt').write_text(text, encoding='utf-8')
        return text

    def get_lexica(self) -> typing.Dict[str, typing.Dict[str, str]]:
        lexica = {}
        for file in self.files:
            dict_ = json.loads(file.read_text(encoding='utf-8'))
            lexica[file.stem] = dict_
        return lexica
