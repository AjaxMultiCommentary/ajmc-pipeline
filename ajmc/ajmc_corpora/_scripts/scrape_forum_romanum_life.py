from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = 'http://www.forumromanum.org/life/'
OUTPUT_DIR = Path('/Users/sven/Desktop/data/forum_romanum/data')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

index_url = BASE_URL + 'johnston.html'

index_soup = BeautifulSoup(requests.get(index_url).text, 'html.parser')
all_text = ''

for td in index_soup.find_all('td', ):
    if td.text.strip().startswith('INTRODUCTION'):
        for link in td.find_all('a', href=True):
            if link.attrs['href'].endswith('.html'):
                page_soup = BeautifulSoup(requests.get(BASE_URL + link.attrs['href']).text, 'html.parser')
                for a in page_soup.find_all('center'):
                    if len(a.text) > 1000:
                        all_text += a.text + '\n'

all_text = all_text.replace('PreviousTable of contentsNext', '')
(OUTPUT_DIR / 'life.txt').write_text(all_text, encoding='utf-8')
