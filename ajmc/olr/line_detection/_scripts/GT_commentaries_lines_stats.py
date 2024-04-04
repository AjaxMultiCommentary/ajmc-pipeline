# This notebook is used to compute the stats of the lines in the GT commentaries lines data

import json

#%% Get the data
from ajmc.commons import variables as vs

via_path = vs.AJMC_DATA_DIR / 'GT-commentaries-lines/GT-commentaries-lines-balanced.json'
data = json.loads(via_path.read_text())

lines = {}
for page_id, page_dict in data['_via_img_metadata'].items():
    filename = page_dict['filename'].split('/')[-1]
    lines[filename] = []
    for region in page_dict['regions']:
        lines[filename].append(region['shape_attributes'])

#%%
commentaries = {p.split('_')[0] for p in lines.keys()}
print(f'A total of {len(commentaries)} have been annotated')
print('Total number of lines:', sum([len(l) for l in lines.values()]))
print('Total number of pages:', len(lines))

#%% Export the datasets stats to a latex table
import pandas as pd

df = pd.DataFrame.from_dict({'Commentary': [p.split('_')[0] for p in lines.keys()],
                             'Pages': [p for p in lines.keys()],
                             'Lines': [len(l) for l in lines.values()]})

#Create a new dataframe with the number of lines and pages per commentary
df = df.groupby('Commentary').agg({'Lines': 'sum', 'Pages': 'nunique'}).sort_values('Commentary')

df['public_domain'] = ['Public domain' if c in vs.PD_COMM_IDS else 'In copyright' for c in df.index]

# Set a multiindex with the public domain column
df.set_index('public_domain', append=True, inplace=True)

# set the index name
df.index.names = ['Commentary', 'Public domain']

# Reorder the multiindex columns so that the public domain column is the first
df = df.reorder_levels(['Public domain', 'Commentary'])

# Sort the index
df.sort_index(inplace=True)

#%%
# Separate the df in two, one for public domain and one for non public domain
df_pd = df.loc['Public domain']
df_npd = df.loc['In copyright']

# Give the dataframes a multiindex with the public domain column
df_pd.index = pd.MultiIndex.from_tuples([('Public domain', c) for c in df_pd.index], names=['', 'Commentary'])
df_npd.index = pd.MultiIndex.from_tuples([('In copyright', c) for c in df_npd.index], names=['', 'Commentary'])

# Add a row with the total number of lines and pages for each dataframe
total_pd = pd.DataFrame({
    "Pages": [df_pd["Pages"].sum()],
    "Lines": [df_pd["Lines"].sum()]
})

# set the index to be the same as the other dataframes
total_pd.index = pd.MultiIndex.from_tuples([('Public domain', 'Total')], names=['', 'Commentary'])
# total_pd.index = ['Public domain']

total_npd = pd.DataFrame({
    # 'Commentary': ['Total'],
    "Pages": [df_npd["Pages"].sum()],
    "Lines": [df_npd["Lines"].sum()]
})

total_npd.index = pd.MultiIndex.from_tuples([('In copyright', 'Total')], names=['', 'Commentary'])
# total_npd.index = ['In copyright']

df_pd = pd.concat([df_pd, total_pd])
df_npd = pd.concat([df_npd, total_npd])

grand_total = pd.DataFrame({
    # 'Commentary': ['Total'],
    "Pages": [df["Pages"].sum()],
    "Lines": [df["Lines"].sum()]})

grand_total.index = pd.MultiIndex.from_tuples([('All', 'Total')], names=['', 'Commentary'])
# grand_total.index = ['All']

#%%
df = pd.concat([df_pd, df_npd, grand_total])

# Reorder the columns
df = df[['Pages', 'Lines']]

styler = df.style

# Bold the font for the total rows
styler.apply(lambda x: ['font-weight: bold' if x.name[1] == 'Total' else '' for i in x], axis=1)

# Export this dataframe to latex
styler.format(escape='latex')

# Hide the index

styler.set_caption("""Line detection dataset statistics.""")

styler.set_table_styles()

latex = styler.to_latex(hrules=True,
                        position_float='centering',
                        convert_css=True,
                        label=f'tab:4_1 {styler.caption}')

import re

latex = re.sub(r' &  & Pages & Lines \\\\\n & Commentary &  &  ', ' & Commentary & Pages & Lines ', latex)
latex = re.sub(r'\n & Total (.*?\n)', r'\n\\cmidrule{2-4} & \\bfseries Total \1\\midrule\n', latex)
latex = re.sub(r'All & Total', r'\\bfseries All & \\bfseries Total', latex)
