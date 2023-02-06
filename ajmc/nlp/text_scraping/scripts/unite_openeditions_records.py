from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

attributes = ['setSpec', 'dcterms:title', 'dcterms:type', 'dcterms:creator', 'dcterms:accessRights', 'dcterms:created',
              'dcterms:publisher', 'dcterms:description', 'dcterms:language', 'dcterms:subject', 'dcterms:spatial',
              'dcterms:temporal', 'dcterms:abstract']

records = {attribute: [] for attribute in attributes}

for path in tqdm(Path('/scratch/sven/openeditions_metadata/qdc_xml').glob('*.xml')):
    # Read an XML file with bs4
    soup = BeautifulSoup(path.read_text('utf-8'), features='xml')

    # find all records elements in soup
    for record in soup.find_all('record'):
        try:
            records['setSpec'].append(record.find('header').find_all('setSpec')[0].text)
        except IndexError:
            records['setSpec'].append(None)

        metadata = record.find('metadata')
        for attribute in attributes[1:]:
            try:
                records[attribute].append(metadata.find(attribute).text)
            except AttributeError:
                records[attribute].append(None)

pd.DataFrame.from_dict(records).to_csv('/scratch/sven/openeditions_metadata/qdc_xml.tsv', sep='\t', index=False)
