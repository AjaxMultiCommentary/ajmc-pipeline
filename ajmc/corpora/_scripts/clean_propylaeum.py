from pathlib import Path

from tqdm import tqdm

from ajmc.commons import unicode_utils as uu
from ajmc.corpora.cleaning_utils import basic_clean

latinised_greek_stopwords = ['kai', 'kal', 'tovto', 'tov', 'yap']
for raw_texts_dir in [Path('/mnt/ajmcdata1/data/propylaeum_BOOKS/raw_texts'), Path('/mnt/ajmcdata1/data/propylaeum_DOK/raw_texts')]:

    total_text = ''
    for txt_path in tqdm(raw_texts_dir.glob('*.txt')):
        text = txt_path.read_text(encoding='utf-8')

        # Check if there are sufficient greek characters
        total_greek_chars = len([c for c in text if uu.get_char_charset(c) == 'greek'])
        unique_greek_chars = len(set([c for c in text if uu.get_char_charset(c) == 'greek']))

        if total_greek_chars > 60 and unique_greek_chars > 10:
            for line in text.split('\n'):
                line = basic_clean(line)
                if len(line) > 100:
                    total_text += line + '\n'

        total_text += '\n\n'

    (raw_texts_dir.parent / 'cleantext.txt').write_text(total_text, encoding='utf-8')
