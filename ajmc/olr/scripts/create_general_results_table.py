import os

import numpy as np
import pandas as pd

from ajmc.commons import variables
from ajmc.olr.layoutlm.layoutlm import create_olr_config
from ajmc.olr.utils import get_olr_splits_page_ids
from ajmc.text_processing.canonical_classes import CanonicalCommentary


def bold_highest_per_xp(df):
    for col in df.columns:
        for row in df.index:
            if df[col][row] == df.loc[row[0]][col].max():
                df[col][row] = "font-weight: bold"
    return df


CONFIGS_DIR = '/scratch/sven/tmp/ajmc/data/layoutlm/configs'

general_results_paths = {
    'LayoutLM': '/scratch/sven/layoutlm/experiments/general_results.tsv',
    'YOLO(BC)+LayoutLM': '/scratch/sven/layoutlm/experiments/general_results_with_yolo_binary.tsv',
    'YOLO(BC)': '/scratch/sven/yolo/runs/binary_class/general_results.tsv',
    'YOLO(MC)': '/scratch/sven/yolo/runs/multiclass/general_results.tsv',
}

# ===================================  READ THE DATAFRAMES   ============================================================
# Read the multiclass dataframes
df = pd.DataFrame()
for model_name, path in general_results_paths.items():
    if model_name == 'YOLO(BC)':
        continue

    temp = pd.read_csv(path, sep="\t", header=[0, 1], index_col=None)
    temp.insert(1, column=('info', 'model'), value=[model_name] * len(temp))
    df = pd.concat([df, temp], axis=0)

# Add the binary data
temp = pd.read_csv(general_results_paths['YOLO(BC)'], sep="\t", header=[0, 1], index_col=None)
temp.insert(1, column=('info', 'model'), value=['YOLO(BC)'] * len(temp))
for col in df.columns:
    if col not in temp.columns:
        temp[col] = [np.nan] * len(temp)
df = pd.concat([df, temp], axis=0, join='inner')


# Set the index
df.set_index(keys=[('info', 'exp'), ('info', 'model')], inplace=True)

#%%
# ===================================  GET NTR AND NTE    ==============================================================
# initialize the columns
for col in df.columns:
    if col[0] not in ['info', 'all']:
        df[(col[0], 'n1_train')] = np.nan
        df[(col[0], 'n2_eval')] = np.nan

#%%
commentaries = {}
for xp_name in df.index.get_level_values(level=0):

    print('****************** Processing', xp_name, '*******************************')
    config = create_olr_config(json_path=os.path.join(CONFIGS_DIR, xp_name + '.json'),
                               prefix=variables.PATHS['cluster_base_dir'])
    # Retrieve the eval pages
    eval_pages = []
    for dict_ in config['data']['eval']:
        if dict_['id'] not in commentaries.keys():
            commentaries[dict_['id']] = CanonicalCommentary.from_json(os.path.join(variables.PATHS['cluster_base_dir'], dict_['id'],
                                                                variables.PATHS['canonical'], dict_['run'] + '.json'))

        commentary = commentaries[dict_['id']]
        page_ids = get_olr_splits_page_ids(commentary.id, [dict_['split']])
        eval_pages += [p for p in commentary.children.pages
                       if p.id in page_ids]

    # Do the eval counts
    eval_stats = {l: 0 for l in config['region_types_to_labels'].values()}

    for p in eval_pages:
        for r in p.children.regions:
            if r.info['region_type'] in config['rois']:
                eval_stats[
                    config['region_types_to_labels'][r.info['region_type']]] += 1

    # Retrieve the train pages
    train_pages = []
    for dict_ in config['data']['train']:
        commentary = CanonicalCommentary.from_json(os.path.join(variables.PATHS['cluster_base_dir'], dict_['id'],
                                                                variables.PATHS['canonical'], dict_['run'] + '.json'))
        page_ids = get_olr_splits_page_ids(commentary.id, [dict_['split']])
        train_pages += [p for p in commentary.children.pages
                        if p.id in page_ids]

    # Do the train counts
    train_stats = {l: 0 for l in config['region_types_to_labels'].values()}

    for p in train_pages:
        for r in p.children.regions:
            if r.info['region_type'] in config['rois']:
                train_stats[
                    config['region_types_to_labels'][r.info['region_type']]] += 1


    # Add the counts to the DF
    for col in df.columns:
        for row in df.index:
            if row[0]==xp_name and col[1]=='n1_train':
                df.loc[row, col]= train_stats[col[0]]
            elif row[0]==xp_name and col[1]=='n2_eval':
                df.loc[row, col]= eval_stats[col[0]]

#%%
# ===================================  GET NTR AND NTE    ==============================================================
# Prettify the df
df.index = pd.MultiIndex.from_tuples([(x[0][:2],x[1]) for x in df.index], names=['Exp', 'Model'])
df.loc[[r for r in df.index if r[1]=='YOLO(BC)'], [c for c in df.columns if c[0] != 'all']] = np.nan
# Round
# for col in df.columns:
#     if col[1] in ['AP', 'P', 'R']:
#         df[col] = df[col].apply(func=lambda x: round(x, ndigits=3))
#     elif col[1] == 'mAP':
#         df[col] = df[col].apply(func=lambda x: round(x, 4))

# set N cols to int
for col in df.columns:
    if col[1] in ['N', 'n1_train', 'n2_eval']:
        df[col] = df[col].astype('Int64')

# Sort
df.sort_index(level=(0, 1), inplace=True)
df.sort_index(axis=1, level=(0, 1), inplace=True)

# create the styler
styler = df.style.format(escape='latex',
                         na_rep='-',
                         precision=2)

styler.set_caption("""General Results table, where bold numbers are applied the highest score in a single experiments. 
N$_{t}$ and N$_{e}$ indicate the counts of instances in train and evaluation set respectively. 
Dashes stand for na-values.""")

styler.hide(axis=0, names=True)
# styler.hide(subset=[(reg, 'N') for reg in df.columns.get_level_values(0) if reg!='all'], axis=1)
styler.hide(subset=[(reg, 'P') for reg in df.columns.get_level_values(0) if reg != 'all'] + \
                   [(reg, 'R') for reg in df.columns.get_level_values(0) if reg != 'all'] + \
                   [(reg, 'N') for reg in df.columns.get_level_values(0) if reg != 'all'] , axis=1)

for xp in set([row[0] for row in styler.index]):
    styler.highlight_max(subset=  #  + \
                         (xp, [('all', 'mAP')]+[(col) for col in df.columns if col[1]=='AP']),
                         props="font-weight: bold")

styler.set_table_styles()

# styler[('0A', 'LayoutLM')][('all', 'mAP')]
latex = styler.to_latex(hrules=True,
                        position_float='centering',
                        convert_css=True,
                        multirow_align='c',
                        multicol_align='c'
                        )

latex = latex.replace(' all ', ' \\textbf{All} ')
latex = latex.replace('app_crit', '\\textbf{App. crit.}')
latex = latex.replace('commentary', '\\textbf{Commentary}')
latex = latex.replace('footnote', '\\textbf{Footnote}')
latex = latex.replace('numbers', '\\textbf{Numbers}')
latex = latex.replace('others', '\\textbf{Others}')
latex = latex.replace('paratext', '\\textbf{Paratext}')
latex = latex.replace('primary_text', '\\textbf{Primary text}')
latex = latex.replace('running_header', '\\textbf{Running h.}')

latex = latex.replace('n1_train', 'N$_{t}$')
latex = latex.replace('n2_eval', 'N$_{e}$')
latex = latex.replace('YOLO(BC)+LayoutLM', 'Y+LLM')
latex = latex.replace('LayoutLM', 'LLM')
latex = latex.replace('YOLO(BC)', 'Y$_{Mono}$')
latex = latex.replace('YOLO(MC)', 'Y$_{Multi}$')
latex = latex.replace('0.', '.')

latex = latex.replace('\\multirow', '\\midrule\n\\multirow')
latex = latex.replace('\\midrule\n\\midrule\n', '\\midrule\n')
latex = latex.replace('\\begin{tabular}', '\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}')
latex = latex.replace('\\end{tabular}', '\\end{tabular}}')

df.to_csv('/scratch/sven/coucou.tsv', sep='\t', )
test = latex