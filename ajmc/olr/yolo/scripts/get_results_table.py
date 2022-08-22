import os
import pandas as pd

runs_path = '/scratch/sven/yolo/runs/binary_classification/'
general_results = pd.DataFrame()

for config_name in sorted(os.listdir(runs_path)):
    if os.path.isdir(os.path.join(runs_path, config_name)):
        results_path = os.path.join(runs_path, config_name, 'results.csv')
        df = pd.read_csv(results_path,header=[0], index_col=None )
        df.columns = pd.Index([c.replace(' ', '') for c in df.columns])
        best = df.iloc[[df['metrics/mAP_0.5'].idxmax()], :]
        best.insert(0, 'exp', [config_name])
        general_results = pd.concat([general_results, best], axis=0)


general_results.to_csv(os.path.join(runs_path, 'general_results.tsv'),
                       sep='\t', index=False)

