import os

import pandas as pd

base_path = '/Users/sven/drive/layout_lm_tests'

results = pd.DataFrame()

for fname in next(os.walk(base_path))[1]:  # Walk in dirs only
    if not fname.startswith('z'):
        best_result = pd.read_csv(os.path.join(base_path, fname, 'results/seqeval/best_results.tsv'),
                                  sep='\t', header=[0, 1])
        best_result.index = [fname]
        results = pd.concat([results, best_result], axis=0)


results.sort_index(axis=0, level=0, inplace=True, key=lambda x: [el[0] for el in x])

results.to_csv('/Users/sven/Desktop/coucou.tsv', sep='\t', )
