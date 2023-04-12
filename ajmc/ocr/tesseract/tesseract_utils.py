import pathlib
import subprocess
from pathlib import Path

import pandas as pd

from ajmc.commons.file_management import get_62_based_datecode
from ajmc.ocr import variables as ocr_vs
from ajmc.ocr.config import CONFIGS


def reformulate_output_dir(output_dir: Path) -> pathlib.Path:
    return output_dir.parent / f'{get_62_based_datecode()}_{output_dir.name}/outputs'


def create_general_table():
    """Creates a general table from the outputs of the tesseract OCR."""

    general_table = pd.DataFrame()
    for config in CONFIGS['experiments'].values():
        xp_dir = ocr_vs.get_experiment_dir(config['id'])
        results_path = xp_dir / 'evaluation' / 'results.txt'
        if not results_path.exists():
            continue
        cer, wer = results_path.read_text('utf-8').split('\n')[1].split('\t')
        config['cer'] = cer
        config['wer'] = wer
        general_table = pd.concat([general_table, pd.DataFrame.from_dict(config)])

    general_table.to_csv(ocr_vs.EXPERIMENTS_DIR / 'general_table.tsv', sep='\t', index=False)

    return general_table


def run_tess_command(command: str):
    """Wrapper around subprocess to run a tesscommand in the proper subshell"""
    command = f'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{ocr_vs.LD_LIBRARY_PATH} ; ' + command
    command = command.encode('ascii')
    result = subprocess.run(['bash'], input=command, shell=True, capture_output=True)
    print(result.stdout.decode('ascii'))
