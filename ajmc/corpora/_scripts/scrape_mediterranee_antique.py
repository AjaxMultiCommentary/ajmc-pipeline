from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


OUTPUT_DIR = Path('/Users/sven/Desktop/data/meditanee_antique/data')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
output_file = OUTPUT_DIR / 'corpus.txt'

INDEX_URLS = ['http://www.mediterranee-antique.fr/Pages_accueil/Accueil_Rome.htm',
              'http://www.mediterranee-antique.fr/Pages_accueil/Accueil_Gaule.htm',
              'http://www.mediterranee-antique.fr/Pages_accueil/Accueil_Grece.htm']

processed = set(INDEX_URLS + ['http://www.mediterranee-antique.fr/index.htm', 'http://www.mediterranee-antique.fr'])
non_valid = set()
external = set()


def get_page_text(url: str):
    try:
        return BeautifulSoup(requests.get(url).content, 'html.parser').text
    except:
        return ''


def skip_url(url: str):
    return any(['mediterranee-antique' not in urlparse(url).netloc,
                url in processed,
                '#' in url,
                'Auteurs/Auteurs/' in url])


for index_url in INDEX_URLS:
    index_soup = BeautifulSoup(requests.get(index_url).content, 'html.parser')
    processed.add(index_url)
    print('NAV: ', index_url)

    for link_tag in index_soup.find_all('a', href=True):
        link_url = urljoin(index_url, link_tag.attrs['href'])
        if skip_url(link_url):
            continue

        processed.add(link_url)

        if link_url.endswith('0.htm'):
            sub_index = BeautifulSoup(requests.get(link_url).content, 'html.parser')
            print('NAV: ', link_url)

            for sub_link_tag in sub_index.find_all('a', href=True):
                sub_link_url = urljoin(link_url, sub_link_tag.attrs['href'])
                if skip_url(sub_link_url):
                    continue

                elif sub_link_url.endswith('.htm'):
                    print('DOW', sub_link_url)
                    output_file.open('a+', encoding='utf-8').write(get_page_text(sub_link_url) + '\n\n\n')
                    processed.add(sub_link_url)
                else:
                    non_valid.add(sub_link_url)

        elif link_url.endswith('.htm'):
            print('DOW: ', link_url)
            output_file.open('a+', encoding='utf-8').write(get_page_text(link_url) + '\n\n\n')
            processed.add(link_url)

        else:
            non_valid.add(link_url)

(OUTPUT_DIR.parent / 'external_links.txt').open('w+', encoding='utf-8').write('\n'.join(external))
(OUTPUT_DIR.parent / 'non_valid_links.txt').open('w+', encoding='utf-8').write('\n'.join(non_valid))
(OUTPUT_DIR.parent / 'processed_links.txt').open('w+', encoding='utf-8').write('\n'.join(processed))
