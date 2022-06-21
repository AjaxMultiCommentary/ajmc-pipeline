import pandas as pd
from ajmc.commons.miscellaneous import read_google_sheet
from ajmc.commons import variables
from ajmc.text_importation.classes import Commentary

df = read_google_sheet(variables.SPREADSHEETS_IDS['olr_gt'], 'olr_gt')
xl_path = '/Users/sven/Desktop/coucou.tsv'
# %%
def get_coarse_type(x):
    coarse_type_mapping = {
        # 'commentary+translation': 'commentary',
        # 'commentary+primary': 'commentary',
        'addenda': 'paratext',
        'appendix': 'paratext',
        'bibliography': 'structured_text',
        'hypothesis': 'paratext',
        'index': 'structured_text',
        'introduction': 'paratext',
        'preface': 'paratext',
        'title': 'structured_text',
        'toc': 'structured_text',
        'translation': 'paratext'}

    return coarse_type_mapping.get(x, x)

df['coarse_layout_type'] = df['layout_type'].apply(get_coarse_type)

df.to_csv(xl_path, sep='\t')

