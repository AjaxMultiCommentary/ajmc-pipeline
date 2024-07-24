"""This is supposed to be run on sampled versions of the JSTOR dataset"""
import gzip
import json
import re
import unicodedata
from pathlib import Path

from tqdm import tqdm

from ajmc.commons import unicode_utils as uu

latinised_greek_stopwords = ['kai', 'kal', 'tovto', 'tov', 'yap']  # removing "to" for english

json_dir = Path('/mnt/ajmcdata1/data/JSTOR-dataset-2021/samples/classical_studies_extended/sampled_jsons')
output_file = Path('/mnt/ajmcdata1/data/JSTOR-dataset-2021/cleaned_corpus.txt')


def remove_running_headers_and_footers2(texts: list) -> list:
    """
    Removes running headers and footers by removing words that are exactly the same at the begining or end of
    at least 2 documents.
    """

    def remove_recurrent_first_words(texts: list) -> list:
        """`texts` is a list of lists of words"""
        finished = False
        t = texts.copy()
        w = 0
        while not finished and any(texts):
            texts = [[token for token in text if token] for text in texts]  # removes empty words
            texts = [text for text in texts if text]  # removes empty pages
            w += 1
            for i, text in enumerate(texts):
                try:
                    if [text_[0] == text[0] for text_ in texts].count(True) >= 2:
                        texts = [text_[1:] if (text_[0] == text[0] and text_[1:]) else text_ for text_ in texts]
                        # texts = [text for text in texts if text]  # removes empty pages
                        break
                    elif [text_[0] == text[0] for text_ in texts].count(True) < 2 and i == len(texts) - 1:
                        finished = True
                except IndexError:
                    print(t, texts)
                    finished = True
                    quit()
                    break
            if w > 200:
                # print(texts)
                break
        return texts

    # Removes numbers within the first 10 words and the last 10 words
    number_cleaned = []
    for text in texts:
        text = ''.join([token for token in text[:10] if not token.isdigit()]) + text[10:-10] + ''.join(
            [token for token in text[-10:] if not token.isdigit()])
        number_cleaned.append(text)

    texts = [text.split() for text in number_cleaned]

    # Removes running headers
    texts = remove_recurrent_first_words(texts)

    # Removes running footers
    texts = remove_recurrent_first_words([list(reversed(text)) for text in texts])
    texts = [list(reversed(text)) for text in texts]

    return [" ".join(text) for text in texts]


def remove_weblinks(text: str) -> str:
    text = re.sub(r'https?://\S+', "", text)
    text = re.sub(r'[a-zA-Z09\.\-_]+@[a-zA-Z09\.\-_]+', "", text)
    return text


def remove_gibberish_pages(texts: list, threshold: float) -> list:
    """Removes pages with a proportion of latin chars inferior to `threshold`"""
    return [text for text in texts if len(re.findall(r'[A-Za-z]', text)) > threshold * len(text)]


def dehyphenate(text: str) -> str:
    text = re.sub(r'(\w+)- (\w+)', r'\1\2', text)
    return text


full_text = ''
for gzip_path in tqdm(sorted(json_dir.glob('*.jsonl.gz'))):
    with gzip.open(gzip_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            doc = data['fullText']
            doc = [unicodedata.normalize('NFC', d) for d in doc]
            doc = remove_running_headers_and_footers2(doc)
            doc = remove_gibberish_pages(doc, 0.5)
            doc = '\n'.join(doc)

            doc = re.sub(r'\s+', ' ', doc)
            doc = re.sub(r'\n+', '\n', doc)
            doc = remove_weblinks(doc)
            doc = dehyphenate(doc)

            greek_words = [word for word in doc.split() if uu.is_charset_string(word, 'greek', 0.9)]
            count_latinised = len([word for word in doc.split() if word in latinised_greek_stopwords])

            if len(greek_words) > 0 and len(set([c for w in greek_words for c in w])) > 6 and count_latinised == 0:
                full_text += doc + '\n\n\n'

output_file.write_text(full_text, encoding='utf-8')
