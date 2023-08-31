from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

INDEX_URL = 'https://remacle.org/'

OUTPUT_DIR = Path('/Users/sven/Desktop/data/remacle_all/data')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
output_file = OUTPUT_DIR / 'all.txt'

processed = {'https://remacle.org/', 'https://remacle.org/'}
non_valid = set()
external = set()


def get_page_text(url: str):
    try:
        return BeautifulSoup(requests.get(url).content, 'html.parser').text
    except:
        return ''


def scrape_recursive(url):
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    processed.add(url)
    print(url)
    for link_tag in soup.find_all('a', href=True):
        sub_url = urljoin(url, link_tag.attrs['href'])
        if sub_url in processed or '#' in sub_url:
            continue

            # INTERNAL LINKS
        if 'remacle.org' in sub_url:
            # TABLE LINKS
            if 'table' in sub_url or 'index' in sub_url:
                scrape_recursive(sub_url)
            # TEXT LINKS
            else:
                processed.add(sub_url)
                print(sub_url)
                text = get_page_text(sub_url)
                if text == '':
                    non_valid.add(sub_url)
                else:
                    output_file.open('a+', encoding='utf-8').write(get_page_text(sub_url) + '\n\n\n')

        # EXTERNAL LINKS
        else:
            external.add(sub_url)


scrape_recursive(INDEX_URL)
(OUTPUT_DIR.parent / 'external_links.txt').open('w+', encoding='utf-8').write('\n'.join(external))
(OUTPUT_DIR.parent / 'non_valid_links.txt').open('w+', encoding='utf-8').write('\n'.join(non_valid))
(OUTPUT_DIR.parent / 'processed_links.txt').open('w+', encoding='utf-8').write('\n'.join(processed))
