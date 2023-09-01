from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def get_page_text(url: str):
    try:
        return BeautifulSoup(requests.get(url).content, 'html.parser').text
    except:
        return ''


OUTPUT_DIR = Path('/Users/sven/Desktop/data/forum_romanum/data')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
output_file = OUTPUT_DIR / 'corpus.txt'

processed = set()

index_urls = ['http://www.forumromanum.org/milne/index.html', 'http://www.forumromanum.org/history/index.html',
              'http://www.forumromanum.org/life/johnston.html']

for index_url in index_urls:
    index_soup = BeautifulSoup(requests.get(index_url).text, 'html.parser')

    for link in index_soup.find_all('a', href=True):
        sub_link = urljoin(index_url, link.attrs['href'])
        if sub_link in processed or '#' in sub_link:
            continue
        if sub_link.endswith('.html'):
            processed.add(sub_link)
            output_file.open('a+', encoding='utf-8').write(get_page_text(sub_link) + '\n\n\n')

(OUTPUT_DIR.parent / 'processed_links.txt').open('w+', encoding='utf-8').write('\n'.join(processed))
