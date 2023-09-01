from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

OUTPUT_DIR = Path('/Users/sven/Desktop/data/agoraclass/data')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
BASE_URL = 'http://agoraclass.fltr.ucl.ac.be/concordances/'
output_file = OUTPUT_DIR / 'corpus.txt'

txts = set()

index_soup = BeautifulSoup(requests.get(BASE_URL + 'intro.htm').text, 'html.parser')

processed = {BASE_URL + 'intro.htm'}

for link in tqdm(index_soup.find_all('a', href=True)):
    sub_url = urljoin(BASE_URL, link.attrs['href'])
    if sub_url in processed or '#' in sub_url:
        continue
    if sub_url.endswith('.txt'):
        txts.add(sub_url)
        continue

    else:
        pass
        text = BeautifulSoup(requests.get(sub_url + '/texte.htm').content, 'html.parser').text

    output_file.open('a+', encoding='utf-8').write(text + '\n\n\n')
    processed.add(sub_url)

(OUTPUT_DIR.parent / 'processed_links.txt').open('w+', encoding='utf-8').write('\n'.join(processed))
