import re
from pathlib import Path

import pandas as pd

metadata_tsv_path = Path('/scratch/sven/scraping/openeditions/metadata/metadata.tsv')

# Read the dataframe
metadata = pd.read_csv(metadata_tsv_path, sep='\t', index_col=False)

metadata_fr = metadata[metadata['dcterms:language'] == 'fr']

fr_keywords = [r'latin', r'grec', r'rome', r'romain', r'antiqu',
               r'philosoph', r'arch[ée]olog', ]


def filter_keywords(x):
    """Filters the dataframe based on the keywords"""

    test = []
    for kw in fr_keywords:
        test.append(bool(re.search(kw, str(x['dcterms:title']))))
        test.append(bool(re.search(kw, str(x['dcterms:description']))))
        test.append(bool(re.search(kw, str(x['dcterms:abstract']))))

    return any(test)


metadata_fr = metadata_fr[metadata_fr.apply(filter_keywords, axis=1)]

#%%
for i, (_, x) in enumerate(metadata_fr.iterrows()):
    print('title:    ', x['dcterms:title'], '\n')
    print('desc:    ', x['dcterms:description'], '\n\n')
    print('abs:    ', x['dcterms:abstract'])
    print()
    if i > 30:
        break
