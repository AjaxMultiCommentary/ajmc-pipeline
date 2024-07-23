from ajmc.commons import variables as vs
from ajmc.commons.file_management import get_ner_spreadsheet


ner_spreadsheet = get_ner_spreadsheet()

#%%
import pandas as pd

# Path to the document selection tsv:
tsv_path = vs.NE_CORPUS_DIR / 'data/preparation/document-selection.tsv'

# Load the document selection tsv, making sure the 'page' column is read as a string:
df = pd.read_csv(tsv_path, sep='\t', dtype={'page': str})

# Make a column concatenating the 'commentary' and 'page' columns:
df['page_id'] = df['commentary'] + '_' + df['page']

#%%
spreadsheet_page_ids = set(ner_spreadsheet['page_id'])
tsv_page_ids = set(df['page_id'])

# make sure the two sets are equal
assert spreadsheet_page_ids == tsv_page_ids

#%% We now check the conformity of the lemlink spreadsheet with the respective git

from ajmc.commons.file_management import get_lemlink_spreadsheet
from ajmc.commons import variables as vs

lemlink_spreadsheet = get_lemlink_spreadsheet()
lemlink_tsv_dir = vs.LEMLINK_CORPUS_DIR / 'data/preparation/corpus/tsv'

# Get the ids present in the spreadsheet from the page_id column and only if `annotated` is True and `in_miniref` is False
ids_in_spreadsheet = set(lemlink_spreadsheet[(lemlink_spreadsheet['annotated'] == True) & (lemlink_spreadsheet['in_miniref'] == False)]['page_id'])

ids_in_dir = set([tsv.stem for tsv in lemlink_tsv_dir.iterdir() if tsv.suffix == '.tsv'])

# See the ids present in the spreadsheet but not in the directory
print('ids in spreadsheet but not in Git')
print(ids_in_spreadsheet - ids_in_dir)

# See the ids present in the directory but not in the spreadsheet
print('ids in Git but not in spreadsheet')
print(ids_in_dir - ids_in_spreadsheet)

#%% We now split the lemlink spreadsheet in train and test sets

# first inspect the dataset
len(lemlink_spreadsheet[lemlink_spreadsheet['annotated'] == True])
#%%
from simple_splitter.split import split

# Split the lemlink spreadsheet in train and test sets

splits = split(splits=[('train', 0.70), ('test', 0.14), ('dev', 0.16)],
               stratification_columns=[lemlink_spreadsheet['commentary_id'].to_list(),
                                       lemlink_spreadsheet['license'].to_list(),
                                       lemlink_spreadsheet['in_miniref'],
                                       lemlink_spreadsheet['annotated']], )

lemlink_spreadsheet['split'] = splits

len(lemlink_spreadsheet[(lemlink_spreadsheet['split'] == 'train') & (lemlink_spreadsheet['annotated'] == True) & (
        lemlink_spreadsheet['commentary_id'] == 'sophoclesplaysa05campgoog')])

#%%

for sp in splits:
    print(sp)
#%%
