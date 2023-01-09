import subprocess
import pathlib
from pathlib import Path
import pandas as pd
from ajmc.ocr import variables as ocr_vars
from ajmc.commons.file_management.utils import get_62_based_datecode
from ajmc.commons.miscellaneous import prefix_command_with_conda_env


def run_tesseract(img_dir: Path,
                  output_dir: Path,
                  langs: str,
                  config: dict = None,
                  psm: int = 3,
                  img_suffix: str = '.png',
                  tessdata_prefix: Path = ocr_vars.TESSDATA_DIR,
                  ):
    """Runs tesseract on images in `img_dir`.

    Note:
        assumes tesseract is installed.

    Args:
        img_dir (Path): path to directory containing images to be OCR'd
        output_dir (Path): path to directory where OCR'd text will be saved
        langs (str): language(s) to use for OCR. Use '+' to separate multiple languages, e.g. 'eng+fra'
        config (dict): dictionary of config options to pass to tesseract. See https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html
        psm (int): page segmentation mode. See https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html
        img_suffix (str): suffix of images to be OCR'd
        tessdata_prefix (Path): path to directory containing tesseract language data
    """


    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the config
    if config:
        (output_dir / 'tess_config').write_text('\n'.join([f'{k} {v}' for k, v in config.items()]), encoding='utf-8')

    command = f"""\
cd {img_dir}; export TESSDATA_PREFIX={tessdata_prefix}; \
for i in *{img_suffix} ; \
do tesseract "$i" "{output_dir}/${{i::${{#i}}-4}}" \
-l {langs} \
--psm {psm} \
{(output_dir /'tess_config') if config else ''}; \
done;"""

    # Writes the command to remember how this was run
    (output_dir / 'command.sh').write_text(command)

    # Write the data related metadata
    if (img_dir / 'metadata.json').is_file():
        (output_dir / 'data_metadata.json').write_bytes((img_dir / 'metadata.json').read_bytes())

    # Run the command
    subprocess.run(['bash'], input=command, shell=True)


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