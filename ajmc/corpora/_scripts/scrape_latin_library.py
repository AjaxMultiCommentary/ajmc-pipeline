from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://www.thelatinlibrary.com/'
OUTPUT_DIR = Path('/Users/sven/Desktop/data/the_latin_library/data')
MENU_LINKS = ['medieval.html', 'index.html', 'classics.html', 'christian.html', 'neo.html', 'humanist.html', 'misc.html']


def get_texts_from_paragraphs(collection_name: str = None, url: str = None, extension: str = '.html', ):
    if url is None:
        soup = BeautifulSoup(requests.get(BASE_URL + collection_name + extension).text, 'html.parser')
    else:
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    paragraphs = soup.find_all('p')
    texts = ''
    for paragraph in paragraphs:
        texts += paragraph.text + '\n'
    (OUTPUT_DIR / f'{collection_name}.txt').write_text(texts, encoding='utf-8')


def get_texts_from_table_index(collection_name: str):
    coll_base_url = BASE_URL + collection_name
    soup = BeautifulSoup(requests.get(BASE_URL + collection_name).text, 'html.parser')
    table = soup.find('table')
    texts = ''
    for row in table.find_all('tr'):
        for cell in row.find_all('td'):
            text_page = requests.get(coll_base_url + '/' + cell.find('a').attrs['href']).text
            text_soup = BeautifulSoup(text_page, 'html.parser')
            for paragraph in text_soup.find_all('p'):
                texts += paragraph.text + '\n'

    (OUTPUT_DIR / f'{collection_name}.txt').write_text(texts, encoding='utf-8')


def get_texts_from_hrefs(collection_name: str, extension: str = '.html'):
    coll_base_url = BASE_URL + collection_name + extension
    soup = BeautifulSoup(requests.get(coll_base_url).text, 'html.parser')
    hrefs = soup.find_all('a', href=True)
    texts = ''
    for href in hrefs:
        if href.attrs['href'] not in MENU_LINKS:
            print(href.attrs['href'])
            sub_url = BASE_URL + href.attrs['href']  # + collection_name + '/'
            sub_soup = BeautifulSoup(requests.get(sub_url).text, 'html.parser')
            for paragraph in sub_soup.find_all('p'):
                texts += paragraph.text + '\n'

    texts = texts.replace('\nThe Latin Library\n', '\n').replace('\nMedieval Latin\n', '\n').replace('\nThe Classics Page\n', '\n').replace(
        '\nChristian Latin\n', '\n')
    (OUTPUT_DIR / f'{collection_name}.txt').write_text(texts, encoding='utf-8')


def scrape_all():
    """Scrapes the entire Latin Library with a bunch of inconsistencies. Check manually."""
    index = BeautifulSoup(requests.get('https://www.thelatinlibrary.com/indices.html').text, 'html.parser')
    table = index.find('table')
    errors = []
    for row in table.find_all('tr'):
        for cell in row.find_all('td'):
            for cell_link in cell.find_all('a', href=True):
                if cell_link.attrs['href'] + '.txt' in OUTPUT_DIR.glob('*.txt'):
                    continue
                print('*****BROWSING {}*****'.format(cell_link.text))
                link = BASE_URL + cell_link.attrs['href']
                link_soup = BeautifulSoup(requests.get(link).text, 'html.parser')
                if any([l.attrs['href'] not in MENU_LINKS for l in link_soup.find_all('a', href=True)]):
                    try:
                        get_texts_from_hrefs(cell_link.attrs['href'].replace('.html', ''))
                    except:
                        errors.append(cell_link.attrs['href'])
                else:
                    try:
                        get_texts_from_paragraphs(collection_name=cell_link.attrs['href'].replace('.html', ''), url=link)
                    except:
                        errors.append(cell_link.attrs['href'])


# get_texts_from_paragraphs('galileo', 'https://www.thelatinlibrary.com/galileo.html')
get_texts_from_hrefs('vitruvius')
