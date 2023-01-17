from pathlib import Path
from typing import Union

import requests
from bs4 import BeautifulSoup


def scrape_page(url):
    # get the html of the page
    html = requests.get(url).text
    # parse the html using beautiful soup and store in variable `soup`
    soup = BeautifulSoup(html, features="xml")
    return soup


def scrape_resumptiontoken_oai(base_request: str,
                               output_dir: Union[Path,str],
                               additional_request_term: str = '',
                               resume:bool = False) -> None:
    """Scrape the resumptiontoken-delimited OAI metadata."""

    output_dir = Path(output_dir)


    if resume:
        # get the last-1 file in the output_dir
        ints = [int(f.stem.split('_')[-1]) for f in output_dir.glob('*.xml')]
        i = max(ints)
        last_file = output_dir / f'data_{i-1}.xml'
        last_soup = BeautifulSoup(last_file.read_text(encoding='utf-8'), features="xml")
        resumption_token = last_soup.find('resumptionToken')

    else:
        # get the first page
        soup = scrape_page(base_request + additional_request_term)
        # get the 'resumptionToken' from the first page
        resumption_token = soup.find('resumptionToken')
        i = 0

    while resumption_token:
        resumption_token = resumption_token.text
        # get the next page
        soup = scrape_page(base_request + '&resumptionToken=' + resumption_token)
        # get the 'resumptionToken' from the next page
        resumption_token = soup.find('resumptionToken')

        # write the soup as an xml file
        (output_dir / f'data_{i}.xml').write_text(str(soup), encoding='utf-8')

        i += 1
