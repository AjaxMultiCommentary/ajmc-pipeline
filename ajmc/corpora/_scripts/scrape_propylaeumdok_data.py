import pandas as pd
import requests
from tqdm import tqdm

from ajmc.corpora import variables as vs

oai_files_dir = vs.BASE_SCRAPE_DIR / 'propylaeum_DOK/metadata/oai_files'
json_output_path = vs.BASE_SCRAPE_DIR / 'propylaeum_DOK/metadata/metadata.json'
output_dir = vs.BASE_SCRAPE_DIR / 'propylaeum_DOK/data'

metadata = pd.read_json(json_output_path, orient='records')
#%%
# FILTER THE METADATA
# filter the metadata to only contain the records with a pdf or text file
filter = metadata['format'].apply(lambda x: 'application/pdf' in x or 'text' in x)
metadata = metadata[filter]

# filter the language
filter = metadata['language'].apply(lambda x: any([l in x for l in ['ger', 'eng', 'fre', 'ita', 'gre', 'lat']]))
metadata = metadata[filter]

#%%
missed = []
for ids in tqdm(metadata['identifier']):
    link = ids[0]
    r = requests.get(link, stream=True)
    path = output_dir / (link.split('/')[-1])
    if r.status_code == 200:
        path.write_bytes(r.content)
    else:
        missed.append(link)

output_dir = (vs.BASE_SCRAPE_DIR / 'propylaeum_DOK/missed.txt').write_text('\n'.join(missed))
