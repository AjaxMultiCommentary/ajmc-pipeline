import subprocess
import pathlib
from pathlib import Path
import pandas as pd
from ajmc.ocr import variables as ocr_vars
from ajmc.commons.file_management.utils import get_62_based_datecode
from ajmc.commons.miscellaneous import prefix_command_with_conda_env


def reformulate_output_dir(output_dir: Path) -> pathlib.Path:
    return output_dir.parent / f'{get_62_based_datecode()}_{output_dir.name}/outputs'


# todo ⚠️ come back here
def create_general_table(xps_dir:Path):
    """Creates a general table from the outputs of the tesseract OCR.

    Args:
        xps_dir (Path): path to the mother directory containing the experiments
    """

    general_table = pd.DataFrame()
    for xp_dir in xps_dir.glob('*'):
        if xp_dir.is_dir():
            xp_name = xp_dir.name
            resizing = [par for par in xp_name.split('_') if 'rsz' in par]
            resizing = int(resizing[0][3:]) if resizing else None
            model = xp_name.split('_')[1]
            psm = [par for par in xp_name.split('_') if 'psm' in par]
            psm = int(psm[0][3:]) if psm else 7

            if (xp_dir / 'evaluation' / 'results.txt').exists():
                with open(xp_dir / 'evaluation' / 'results.txt', 'r') as f:
                    cer, wer = f.read().split('\n')[1].split('\t')
                general_table = pd.concat([general_table,
                                           pd.DataFrame({'xp_name': [xp_name],
                                                         'models': [model],
                                                         'data': [None],
                                                         'psm': [psm],
                                                         'resize': [resizing],
                                                         'cer': [cer],
                                                         'wer': [wer]})
                                           ])

    general_table.to_csv(xps_dir / 'general_table.tsv', sep='\t', index=False)
    return general_table



# Todo see if this is still necesary
def prefix_tess_command(command:str, env_name = ocr_vars.CONDA_ENV, conda_install_dir=ocr_vars.CONDA_INSTALL_DIR):
    command = 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LD_LIBRARY_PATH/lib/:$CONDA_PREFIX/lib/ ;' + command
    return prefix_command_with_conda_env(command, env_name=env_name, conda_install_dir=conda_install_dir)


# Todo see if this is still necesary
def run_tess_command(command:str, env_name = ocr_vars.CONDA_ENV, conda_install_dir=ocr_vars.CONDA_INSTALL_DIR):
    """Wrapper around subprocess to run a tesscommand in the proper subshell"""
    command = prefix_tess_command(command, env_name=env_name, conda_install_dir=conda_install_dir)
    command = command.encode('ascii')
    return subprocess.run(['bash'], input=command, shell=True, capture_output=True)