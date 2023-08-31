import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = 'http://khazarzar.skeptik.net/pgm/PG_Migne/'
OUTPUT_DIR = Path('/Users/sven/Desktop/data/patrologia_graeca/data')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

metadata = {'languages': ['gre']}
(OUTPUT_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=4), encoding='utf-8')


def walk_patrologia_graeca(dir_url):
    soup = BeautifulSoup(requests.get(dir_url).text, 'html.parser')
    links = soup.find_all('a', href=True)
    for link in links:
        if link.attrs['href'].endswith('/') and link.text != 'Parent Directory':
            walk_patrologia_graeca(dir_url + link.attrs['href'])
        elif link.attrs['href'].endswith('.pdf'):
            (OUTPUT_DIR / link.attrs['href'].replace('%20', '_')).write_bytes(requests.get(dir_url + link.attrs['href']).content)


walk_patrologia_graeca(BASE_URL)
