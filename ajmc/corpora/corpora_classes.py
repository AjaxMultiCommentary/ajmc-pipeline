import json
import re
import shutil
import typing
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import beta_code
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from tqdm import tqdm

from ajmc.commons.unicode_utils import is_charset_string, get_char_charset
from ajmc.corpora import variables as vs
from ajmc.corpora.cleaning_utils import harmonise_linebreaks


class Corpus(ABC):
    """Mother class for all Corpus-objects. Can be used to instantiate child-object via ``auto_init``"""

    def __init__(self, corpus_id: str):
        self.id = corpus_id
        self.root_dir = vs.ROOT_STORING_DIR / corpus_id
        self.metadata_path = self.root_dir / 'metadata.json'
        self.metadata = json.loads(self.metadata_path.read_text(encoding='utf-8'))


    @classmethod
    def auto_init(cls, corpus_id):
        """This classmethod can be used to create the right object from the mother class"""
        metadata_path = vs.ROOT_STORING_DIR / corpus_id / 'metadata.json'
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
        """Chunks the corpus into strings of ``chunk-size`` units.

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


    # Todo
    def chunk_to_sentences(self, max_length: int = 2048):
        raise NotImplementedError

    def write_plain_text(self, path: typing.Optional[typing.Union[str, Path]] = None):
        if path is None:
            (self.data_dir / 'corpus.txt').write_text(self.get_plain_text(), encoding='utf-8')
        else:
            Path(path).write_text(self.get_plain_text(), encoding='utf-8')


    @staticmethod
    def is_valid_greek(text: str, stopwords_threshold: int = 2, greek_chars_threshold: float = 0.001) -> bool:
        total_greek_chars = len([c for c in text if get_char_charset(c) == 'greek'])
        latinised_greek_stopwords = ['kai', 'kal', 'tovto', 'tov', 'yap']

        # if total_greek_chars/len(text) < 0.01:
        stopwords_count = sum([1 for word in text.split() if word.lower() in latinised_greek_stopwords])
        greek_chars_ratio = total_greek_chars / len(text)

        # Case 1: Low Greek chars, low stopwords
        if greek_chars_ratio < greek_chars_threshold and stopwords_count < stopwords_threshold:
            return True
        # Case 2: Low Greek chars, high stopwords
        elif greek_chars_ratio < greek_chars_threshold and stopwords_count >= stopwords_threshold:
            return False
        # Case 3: High Greek chars, low stopwords
        elif greek_chars_ratio >= greek_chars_threshold and stopwords_count < stopwords_threshold + 10:
            return True
        else:
            return False

class PlainTextCorpus(Corpus):

    def __init__(self, corpus_id: str):
        super().__init__(corpus_id)

    @property
    def data_dir(self) -> Path:
        return self.root_dir / 'data'

    @property
    def files(self) -> typing.List[Path]:
        return [self.root_dir / 'cleantext.txt']

    def get_plain_text(self) -> str:
        return self.files[0].read_text(encoding='utf-8')


class PdfCorpus(PlainTextCorpus):
    def __init__(self, corpus_id: str):
        super().__init__(corpus_id)
        self.raw_texts_dir = self.root_dir / 'raw_texts'
        self.data_dir.mkdir(exist_ok=True)
        self.raw_texts_dir.mkdir(exist_ok=True)
        self.pdf_dir = self.root_dir / 'pdfs'

    @staticmethod
    def parse_pdf(pdf_path: Path):
        """Parses a pdf file and returns the text, using pdfminer.six.

        Warning:
            This function does not handle the extraction of greek characters properly all the time.

        Args:
            pdf_path: The path to the pdf file to parse
        """
        text = ''
        for page_layout in extract_pages(pdf_path):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text += element.get_text() + '\n'
        return text

    def extract_texts(self, **kwargs):
        """Extracts the text from the pdfs and writes it to files in the raw_texts_dir"""
        for pdf_path in tqdm(sorted(self.pdf_dir.rglob('*.pdf')), desc='Extracting texts', total=len(list(self.pdf_dir.rglob('*.pdf')))):
            if (self.raw_texts_dir / (pdf_path.stem + '.txt')).exists():
                continue
            try:
                text = self.parse_pdf(pdf_path)
                (self.raw_texts_dir / (pdf_path.stem + '.txt')).write_text(text, encoding='utf-8')
            except Exception as e:
                print(f'Error while parsing {pdf_path}: {e}')
                continue


    def clean_texts(self, stopwords_threshold: int = 2, greek_chars_threshold: float = 0.001):

        total_text = ''
        deleted = 0
        for txt in tqdm(self.raw_texts_dir.glob('*.txt')):
            text = txt.read_text(encoding='utf-8')
            if not self.is_valid_greek(text, stopwords_threshold, greek_chars_threshold):
                deleted += 1
            else:
                total_text += text + '\n\n\n'

        print(f'{deleted} documents deleted')
        (self.root_dir / 'cleantext.txt').write_text(total_text, encoding='utf-8')


class BrillCorpus(PdfCorpus):

    def __init__(self, corpus_id: str):
        super().__init__(corpus_id)

    def extract_texts(self, source_dir: Path, tmp_dir: Path):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(exist_ok=True)
        self.raw_texts_dir.mkdir(exist_ok=True)

        # Read a zip file
        for zip_path in tqdm(list(source_dir.glob('*.zip')), desc='Extracting texts'):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)

            for sub_zip_path in tmp_dir.rglob('*.zip'):
                print('Subzips found:', sub_zip_path)
                sub_zip_dir = tmp_dir / sub_zip_path.stem
                sub_zip_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(sub_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(sub_zip_dir)

            text = ''
            for pdf_path in sorted(tmp_dir.rglob('*.pdf')):
                for page_layout in extract_pages(pdf_path):
                    for element in page_layout:
                        if isinstance(element, LTTextContainer):
                            text += element.get_text() + '\n'

            (self.raw_texts_dir / (zip_path.stem + '.txt')).write_text(text, encoding='utf-8')

            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(exist_ok=True)


class TeiCorpus(Corpus):

    def __init__(self, corpus_id: str):
        super().__init__(corpus_id)

    @property
    def data_dir(self) -> Path:
        return self.root_dir / 'data'

    @property
    def files(self) -> typing.List[Path]:
        return list(self.data_dir.glob('*.xml'))

    def get_plain_text(self) -> str:
        if (self.root_dir / 'cleantext.txt').exists():
            return (self.root_dir / 'cleantext.txt').read_text(encoding='utf-8')

        full_text = ''
        for file in self.files:
            soup = BeautifulSoup(file.read_text(encoding='utf-8'), features='xml')
            for text in soup.find_all(re.compile(r'(?:tei:)?text')):
                full_text += text.text + '\n\n\n'

        (self.root_dir / 'cleantext.txt').write_text(full_text, encoding='utf-8')
        return full_text


class OGLCorpus(TeiCorpus):

    def __init__(self, corpus_id: str):
        super().__init__(corpus_id)

    @property
    def data_dir(self) -> Path:
        return self.root_dir / self.id / 'data'

    @property
    def files(self) -> typing.List[Path]:
        return [p for p in self.data_dir.rglob('*.xml') if p.name[0] != '_']


class PerseusLegacyCorpus(Corpus):


    def __init__(self, corpus_id: str):
        super().__init__(corpus_id)

    @property
    def data_dir(self) -> Path:
        return self.root_dir / 'data'

    @property
    def files(self) -> typing.List[Path]:
        return list(self.data_dir.rglob('*.xml'))


    def get_plain_text(self) -> str:
        if (self.root_dir / 'cleantext.txt').exists():
            return (self.root_dir / 'cleantext.txt').read_text(encoding='utf-8')

        text = '\n\n\n'.join([self.read_document(file) for file in self.files])
        text = harmonise_linebreaks(text)
        (self.root_dir / 'cleantext.txt').write_text(text, encoding='utf-8')
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
        return self.root_dir / 'data'

    @property
    def files(self) -> typing.List[Path]:
        return list((self.data_dir / 'cleaned').rglob('*.json'))


    def get_plain_text(self) -> str:
        if (self.root_dir / 'cleantext.txt').exists():
            return (self.root_dir / 'cleantext.txt').read_text(encoding='utf-8')
        text = ''
        for file in self.files:
            dict_ = json.loads(file.read_text(encoding='utf-8'))
            text += '\n'.join([v if v.endswith('.') else v + '.' for v in dict_.values()]) + '\n\n\n'

        (self.root_dir / 'cleantext.txt').write_text(text, encoding='utf-8')
        return text

    def get_lexica(self) -> typing.Dict[str, typing.Dict[str, str]]:
        lexica = {}
        for file in self.files:
            dict_ = json.loads(file.read_text(encoding='utf-8'))
            lexica[file.stem] = dict_
        return lexica


class EpibauCorpus(Corpus):

    def __init__(self, corpus_id: str = 'EpibauCorpus'):
        super().__init__(corpus_id)

    @property
    def data_dir(self) -> Path:
        return self.root_dir / 'data/release/v0.3/'

    @property
    def files(self) -> typing.List[Path]:
        return [p for p in self.data_dir.rglob('*.tsv') if 'masked' not in p.name]


    def get_plain_text(self) -> str:
        if (self.root_dir / 'cleantext.txt').exists():
            return (self.root_dir / 'cleantext.txt').read_text(encoding='utf-8')
        text = ''
        for file in self.files:
            text += ' '.join([l.split('\t')[0] for l in file.read_text(encoding='utf-8').splitlines()[1:]])
        (self.root_dir / 'cleantext.txt').write_text(text, encoding='utf-8')
        return text
