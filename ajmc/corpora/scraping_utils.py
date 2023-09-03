"""Tools for online text webscraping."""

from pathlib import Path
from typing import Union, List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def get_html(url):
    return requests.get(url).text


def scrape_page(url):
    # get the html of the page
    request = requests.get(url)
    request.encoding = 'utf-8'
    # parse the html using beautiful soup and store in variable ``soup``
    soup = BeautifulSoup(request.text, features="xml", )
    return soup


def scrape_resumptiontoken_oai(base_request: str,
                               output_dir: Union[Path, str],
                               additional_request_term: str = '',
                               resume: bool = False,
                               filename_prefix: str = 'data') -> None:
    """Scrape the resumptiontoken-delimited OAI metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        # get the last-1 file in the output_dir
        ints = [int(f.stem.split('_')[-1]) for f in output_dir.glob('*.xml')]
        i = max(ints)
        last_file = output_dir / f'data_{i - 1}.xml'
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
        (output_dir / f'{filename_prefix}_{i}.xml').write_text(str(soup), encoding='utf-8')

        i += 1


def oai_files_to_dataframe(attributes: List[str],
                           oai_files_dir: Path,
                           json_output_path: Path,
                           extension: str = '.xml',
                           attribute_prefix: str = 'dcterms:',
                           ) -> pd.DataFrame:
    records = {attribute: [] for attribute in ['setSpec'] + attributes}

    for path in tqdm(oai_files_dir.glob(f'*{extension}')):
        # Read an XML file with bs4
        soup = BeautifulSoup(path.read_text('utf-8'), features='xml')

        # find all records elements in soup
        for record in soup.find_all('record'):
            header = record.find('header')
            metadata = record.find('metadata')

            if metadata is None or header is None:
                continue

            records['setSpec'].append([s.text for s in header.find_all('setSpec')])

            for attribute in attributes:
                records[attribute].append([s.text for s in metadata.find_all(attribute_prefix + attribute)])

    df = pd.DataFrame.from_dict(records)
    df.to_json(json_output_path, force_ascii=False, orient='records')

    return df
