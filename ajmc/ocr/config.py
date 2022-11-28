"""This module handles reading of configs from excel."""
import json

import pandas as pd
from pathlib import Path

xl_path = Path('/Users/sven/drive/sven_ocr_xps.xlsx')

df = pd.read_excel(xl_path, sheet_name='datasets', keep_default_na=False)

# %%

df_row = df.iloc[0]


# %%

def df_row_to_dataset_config(df_row: 'pd.Series'):
    config = df_row.to_dict()

    # Replace '' with None
    for k, v in config.items():
        if v == '':
            config[k] = None

    # split composed values
    for par, val in config.items():
        if type(val) == str and val != 'id':
            config[par] = set(val.split('+'))  # todo ⚠️ this is assume there is no '+' in the sources !

    return config



#%%

def get_dataset_path(config: dict,
                     parent_dir: Path):
    """Returns path to a dataset, creating it if it doesn't exist"""

    for config_path in parent_dir.rglob('config.json'):
        config_ = json.loads(config_path.read_text())
        if config_ == config:
            return config_path.parent

    create_dataset_from_config(config, parent_dir)
    return parent_dir / config['id']




