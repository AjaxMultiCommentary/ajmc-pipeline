"""This is a simple script to prepare and clean and clean gregs pdlrefwk repository (https://github.com/gregorycrane/canonical_pdlrefwk)
 and make it ready for ajmc's processing. """
from pathlib import Path

import pandas as pd

base_dir = Path('/mnt/ajmcdata1/data/canonical_pdlrefwk-master')

# Rename the directory
new_dir = '/mnt/ajmcdata1/data/perseus_secondary'
base_dir = base_dir.rename(Path(new_dir))
base_dir = Path(new_dir)
(base_dir / 'pdlrefwk').rename(base_dir / 'data')

#%% Change the readme
readme_path = base_dir / 'README.md'
readme_text = readme_path.read_text(encoding='utf-8')
readme_table = [l for l in readme_text.split('\n') if l.startswith('|')]
readme_table = [l.split('|')[1:-1] for l in readme_table]
readme_table = [[c.strip() for c in l] for l in readme_table]
readme_table.pop(1)
readme_table = pd.DataFrame(readme_table[1:], columns=readme_table[0])

index_path = base_dir / 'index.tsv'
readme_table.to_csv(index_path, sep='\t', index=False)
