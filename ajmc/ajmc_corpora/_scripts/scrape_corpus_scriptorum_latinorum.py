from pathlib import Path

import requests
from bs4 import BeautifulSoup


BASE_URL = 'http://www.forumromanum.org/literature/'
OUTPUT_DIR = Path('/Users/sven/Desktop/data/corpus_scriptorum_latinorum/data')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

output_file = OUTPUT_DIR / 'corpus_scriptorum_latinorum.txt'

already_done = []
for letters in ['a', 'b-d', 'e-g', 'h-l', 'm-o', 'p-r', 's-t', 'u-z']:
    index_url = BASE_URL + 'authors_' + letters + '.html'
    index_soup = BeautifulSoup(requests.get(index_url).text, 'html.parser')
    text = ''
    for td in index_soup.find_all('td'):
        if td.text.strip().startswith('Browse by'):
            for author_link in td.find_all('a', href=True):
                if author_link.attrs['href'].startswith('author') or author_link.attrs['href'] in already_done:
                    continue
                print(author_link.attrs['href'])
                authors_soup = BeautifulSoup(requests.get(BASE_URL + author_link.attrs['href']).text, 'html.parser')

                try:
                    main_tag = [tag for tag in authors_soup.find_all('td') if 'Works' in tag.text][0]
                except IndexError:
                    continue

                for text_link in main_tag.find_all('a', href=True):
                    if any([pat in text_link.attrs['href'] for pat in ['http', 'www.', '.com']]) or text_link.attrs['href'] in already_done:
                        continue
                    if text_link.attrs['href'].endswith('.html'):
                        text_soup = BeautifulSoup(requests.get(BASE_URL + text_link.attrs['href']).text, 'html.parser')
                        output_file.open('a+', encoding='utf-8').write(text_soup.text + '\n')
                    elif text_link.attrs['href'].endswith('.txt'):
                        output_file.open('a+', encoding='utf-8').write(requests.get(BASE_URL + text_link.attrs['href']).text + '\n')
                    already_done.append(text_link.attrs['href'])
                already_done.append(author_link.attrs['href'])
