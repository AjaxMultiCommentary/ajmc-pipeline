from pathlib import Path

corpus_ids = [
    'Brill-KIEM-data',
    'JSTOR-dataset-2021',
    'wiki_la',
    'wiki_en',
    'wiki_it',
    'wiki_fr',
    'wiki_el',
    'wiki_de',
    'riemenschneider_born_digital',
    'riemenschneider_internet_archive',
    'propylaeum_BOOKS',
    'propylaeum_DOK',
]

root_dir = Path('/mnt/ajmcdata1/data')
for corpus_id in corpus_ids:
    text = (root_dir / corpus_id / 'cleantext.txt').read_text(encoding='utf-8')
    print(len(text))
