import pandas as pd
from ajmc.commons.miscellaneous import read_google_sheet
from ajmc.commons import variables
from ajmc.text_importation.classes import Commentary



sheet_id = '1_hDP_bGDNuqTPreinGS9-ShnXuXCjDaEbz-qEMUSito'

# Do the loop to store all sub-sheets in a single dict
dfs = {}
for id_ in variables.COMMENTARY_IDS:
    if id_ not in ['Finglass2011', 'lestragdiesdeso00tourgoog']:
        df = read_google_sheet(sheet_id, id_)
        df['id'] = id_
        df['page_id'] = df['image number'].apply(lambda x: id_ + '_' + (4 - len(str(x))) * '0' + str(x))

        # Verify if this is coherent with the existing via_project
        commentary = Commentary.from_folder_structure(commentary_id=id_)
        olr_full_gt_ids = []
        olr_comm_only = []
        for k, v in commentary.via_project['_via_img_metadata'].items():
            if any([r['region_attributes']['text'] not in ['commentary', 'undefined'] for r in v['regions']]):
                olr_full_gt_ids.append(v['filename'].split('.')[0])

            elif all([r['region_attributes']['text'] in ['commentary', 'undefined'] for r in v['regions']]) and \
                    any([r['region_attributes']['text'] in ['commentary'] for r in v['regions']]):
                olr_comm_only.append(v['filename'].split('.')[0])

        all_annotated_via = set(olr_full_gt_ids)  # + olr_comm_only
        all_annotated_sheet = set(df['page_id'])

        if all_annotated_sheet.difference(all_annotated_via):
            print(f'Running {id_} - The following pages are in annotated in sheet but not in via : ')
            print(all_annotated_sheet.difference(all_annotated_via))
            print('')

        if all_annotated_via.difference(all_annotated_sheet):
            print(f'Running {id_} - The following pages are in annotated in via but not in sheet : ')
            print(all_annotated_via.difference(all_annotated_sheet))
            print('')

        df.sort_values('image number', inplace=True)
        # Add `df` to `dfs`
        dfs[id_] = df

# %% Concatenate all dfs
df_export = pd.DataFrame()
for id, df in dfs.items():
    df_export = pd.concat([df_export, df], axis=0, ignore_index=True)
assert len(df_export) == sum([len(df) for df in dfs.values()])


# %% Unify types

def unify_types(x):
    correct_types_mapping = {'english commentary': 'commentary+translation',
                             'commentary_latin': 'commentary+translation',
                             'commentary_italian': 'commentary+translation',
                             'commentary_english': 'commentary+translation',
                             'greek commentary': 'commentary+primary',
                             'commentary_greek': 'commentary+primary',
                             'prefatio': 'preface',
                             'table_of_contens': 'TOC'}

    return correct_types_mapping.get(x, x)


df_export['type'] = df_export['type'].apply(unify_types)

set(df_export['type'])

# reorder columns
df_export = pd.concat([df_export['id'],
                       df_export['page_id'],
                       df_export['image number'],
                       df_export['usage'],
                       df_export['annotator'],
                       df_export['notes'],
                       df_export['type'],
                       df_export['status']
                       ], axis=1)

# %%


# %%
xl_path = '/Users/sven/Desktop/coucou.tsv'
df_export.to_csv(xl_path, sep='\t')

# %% Import re_created df and do the splitting.
from ajmc.commons.miscellaneous import read_google_sheet
sheet_id = '1_hDP_bGDNuqTPreinGS9-ShnXuXCjDaEbz-qEMUSito'
df = read_google_sheet(sheet_id, 'olr_gt')

set(df['layout_type'])
xl_path = '/Users/sven/Desktop/coucou.tsv'


# %%
def get_coarse_type(x):
    correct_types_mapping = {'commentary+translation': 'commentary',
                             'commentary+primary': 'commentary',
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

    return correct_types_mapping.get(x, x)


df['coarse_layout_type'] = df['layout_type'].apply(get_coarse_type)
df.to_csv(xl_path, sep='\t')
set(df['coarse_layout_type'])

# %% Train, Dev, Test - split
import numpy as np
import pandas as pd

splitting = {'train': 0.7, 'dev': 0.15, 'test': 0.15}

dfs = []
split_array = []
for id in set(df['id']):
    for type in set(df['coarse_layout_type']):
        df_temp = df[(df['id'] == id) & (df['coarse_layout_type'] == type)]

        sets = {
            k: v for v, k in zip(
                np.split(df_temp,
                         [int(splitting['train'] * len(df_temp)),  # we set test first to prioritize entries
                          int((splitting['train'] + splitting['dev']) * len(df_temp))]
                         ),
                list(splitting.keys()
                     ))
        }

        for split in sets.keys():
            sets[split]['split']=split

        df_append = pd.concat([d for d in sets.values()])
        assert len(df_append) == len(df_temp)
        dfs.append(df_append)


dfnew = pd.concat(dfs)
assert len(dfnew)==len(df)
assert set(dfnew['page_id']) == set(df['page_id'])

dfnew.to_csv(xl_path, sep='\t')
# %%

