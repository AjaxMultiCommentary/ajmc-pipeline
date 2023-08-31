import re

import requests
from tqdm import tqdm

from ajmc.ajmc_corpora import variables as vs

metadata_dir = vs.BASE_SCRAPE_DIR / 'propylaeum_BOOKS/metadata'
output_dir = vs.BASE_SCRAPE_DIR / 'propylaeum_BOOKS/data'
catalog_base_url = 'https://books.ub.uni-heidelberg.de/propylaeum/catalog/book/'
download_base_url = 'https://books.ub.uni-heidelberg.de'

for path in tqdm(metadata_dir.glob('*.html'), total=len(list(metadata_dir.glob('*.html')))):

    text = path.read_text('utf-8')
    result = re.findall(r'(?<=li role="presentation"><a href=")/propylaeum/reader/download/.*\.pdf', text)
    if len(result) == 1:
        if (output_dir / result[0].split('/')[-1]).exists():
            continue
        r = requests.get(download_base_url + result[0])
        (output_dir / result[0].split('/')[-1]).write_bytes(r.content)
    else:
        for result in re.findall(r'/propylaeum/reader/download/.*\.pdf', text):
            r = requests.get(download_base_url + result)
            (output_dir / result.split('/')[-1]).write_bytes(r.content)
