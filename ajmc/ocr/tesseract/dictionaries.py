"""Utils to handle tesseract dictionaries, BR and MR's dictionaries and combine them with tesseract's dictionaries
to create new models.

Architecture should be:
```
XP_DIR / models / model_name / model_name.traineddata
```
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.ocr import variables as ocr_vars
import pandas as pd

from ajmc.ocr.utils import is_greek_string, is_latin_string, is_number_string
from ajmc.commons.miscellaneous import prefix_command_with_conda_env
from ajmc.ocr.tesseract.tesseract_utils import run_tess_command

logger = get_custom_logger(__name__)


def merge_wordlists(*wordlists: List[str]) -> List[str]:
    """Merge wordlists"""
    wordlist = set()
    for i, wl in enumerate(wordlists):
        logger.info(f"Adding word list {i} with {len(wl)} words.")
        wordlist.update(wl)
    logger.info(f"Final word list has {len(wordlist)} words.")
    return list(wordlist)


def get_mr_abbr_wordlist(langs: List[str]) -> List[str]:
    """Get the wordlist of the MR abbreviation dictionary"""
    langs += ['default', 'abbr']

    # Get MR abbreviation dictionary
    dfa = pd.read_csv(ocr_vars.DICTIONARIES_DIR / 'mr_authors.tsv', encoding='utf-8', sep='\t',
                      keep_default_na=False)
    dfw = pd.read_csv(ocr_vars.DICTIONARIES_DIR / 'mr_works.tsv', encoding='utf-8', sep='\t',
                      keep_default_na=False)
    df = pd.concat([dfa, dfw], axis=0)

    # Define a filter
    def filter_func(x):
        if "gre" in langs:
            return x['TYPE'] in langs and is_greek_string(x['TEXT'], threshold=0.9)
        else:
            return x['TYPE'] in langs and is_latin_string(x['TEXT'], threshold=0.9)

    # Creates the Filter
    filter_ = df.apply(filter_func, axis=1)

    # Apply the filter
    abbreviations = df[filter_]['TEXT'].tolist()

    # One word per line
    abbreviations = list(set(' '.join(abbreviations).split()))

    # Delete numbers
    abbreviations = [x for x in abbreviations if not is_number_string(x, threshold=0.9)]

    return abbreviations


def write_mr_abbr_wordlist(langs: List[str]):
    """Write the wordlist of the MR abbreviation dictionary"""
    wordlist = get_mr_abbr_wordlist(langs)
    file_name = f'mr{langs[0]}.txt'
    (ocr_vars.DICTIONARIES_DIR / file_name).write_text('\n'.join(wordlist), encoding='utf-8')


def write_all_wordlists():
    """Write all wordlists"""
    write_mr_abbr_wordlist(['gre'])
    write_mr_abbr_wordlist(['lat'])
    write_mr_abbr_wordlist(['eng'])
    write_mr_abbr_wordlist(['ita'])
    write_mr_abbr_wordlist(['fre'])
    write_mr_abbr_wordlist(['ger'])
    write_mr_abbr_wordlist(['spa'])

    for path in ocr_vars.MODELS_DIR.rglob('*.traineddata'):
        wordlist = get_traineddata_wordlist(path.stem)
        file_name = f'{path.stem}.txt'
        (ocr_vars.DICTIONARIES_DIR / file_name).write_text('\n'.join(wordlist), encoding='utf-8')


def get_traineddata_wordlist(traineddata_name) -> List[str]:
    """Get the wordlist of a traineddata file"""

    write_unpacked_traineddata(ocr_vars.get_trainneddata_path(traineddata_name))
    unpacked_dir = ocr_vars.get_traineddata_unpacked_dir(traineddata_name)
    # Get wordlist
    dawg_path = unpacked_dir / f'{traineddata_name}.lstm-word-dawg'
    unicharset_path = unpacked_dir / f'{traineddata_name}.lstm-unicharset'
    wordlist_path = ocr_vars.DICTIONARIES_DIR / f'{traineddata_name}.txt'
    command = f'dawg2wordlist {unicharset_path} {dawg_path} {wordlist_path}'
    run_tess_command(command)

    return wordlist_path.read_text(encoding='utf-8').splitlines()


def get_or_create_wordlist_path(wordlist_name: str) -> Path:
    """Gets the path to wordlist.txt file, creating it if it doesn't exist"""

    wordlist_path = ocr_vars.get_wordlist_path(wordlist_name)
    if not wordlist_path.is_file():
        final_list = merge_wordlists([(ocr_vars.DICTIONARIES_DIR / (l + '.txt')).read_text(encoding='utf-8').splitlines()
                                      for l in wordlist_name.split(ocr_vars.SEPARATOR)])
        wordlist_path.write_text('\n'.join(final_list), encoding='utf-8')

    return wordlist_path


def change_traineddata_wordlist(traineddata_name: str, wordlist_name: str):
    """Change the wordlist of a traineddata file"""

    traineddata_path = ocr_vars.get_trainneddata_path(traineddata_name)
    write_unpacked_traineddata(traineddata_path)

    # Get the path to the unpacked directory
    unpacked_dir = ocr_vars.get_traineddata_unpacked_dir(traineddata_name)
    wordlist_path = get_or_create_wordlist_path(wordlist_name)

    # Copy the wordlist
    command = f'wordlist2dawg {wordlist_path} {unpacked_dir}/{traineddata_name}.lstm-unicharset {unpacked_dir}/{traineddata_name}.lstm-word-dawg'
    run_tess_command(command)

    # Pack the traineddata
    command = f'combine_tessdata {unpacked_dir}/{traineddata_name}.'
    run_tess_command(command)

    # Remove the old traineddata
    traineddata_path.unlink()

    # move the new traineddata
    command = f'mv {unpacked_dir}/{traineddata_name}.traineddata {traineddata_path.parent}'
    subprocess.run(['bash'], input=command.encode('ascii'), shell=True, capture_output=True)

    # Delete the unpacked directory
    # command = f'rm -r {unpacked_dir}'
    # os.system(command)


def create_traineddata_with_wordlist(source_traineddata_name,
                                     wordlist_names: List[str],
                                     output_model_name: Optional[str] = None):
    if output_model_name is None:
        output_model_name = (source_traineddata_name + '_' + ','.join(wordlist_names))

    # Create the models repo
    output_model_dir = Path(ocr_vars.MODELS_DIR / output_model_name)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ocr_vars.get_trainneddata_path(source_traineddata_name), output_model_dir)

    wordlists = [(ocr_vars.DICTIONARIES_DIR / f'{wordlist_name}.txt').read_text('utf-8').splitlines()
                 for wordlist_name in wordlist_names]

    # merge the two list
    merged_wordlist = merge_wordlists(*wordlists)

    # write the new wordlist
    merged_wordlist_name = ','.join(wordlist_names)
    (ocr_vars.DICTIONARIES_DIR / f'{merged_wordlist_name}.txt').write_text('\n'.join(merged_wordlist),
                                                                           encoding='utf-8')

    # Change the traineddata's wordlist
    change_traineddata_wordlist(traineddata_name=output_model_name,
                                wordlist_name=merged_wordlist_name)


def write_unpacked_traineddata(traineddata_path: Path,
                               unpacked_dir: Path = None):
    """Unpacks a traineddata file"""

    # Set path to custom data
    if unpacked_dir is None:
        unpacked_dir = ocr_vars.get_traineddata_unpacked_dir(traineddata_path.stem)

    unpacked_dir.mkdir(parents=True, exist_ok=True)

    # Unpack traineddata
    command = f'combine_tessdata -u {traineddata_path} {unpacked_dir}/{traineddata_path.stem}.'
    run_tess_command(command)


def write_combined_wordlists():
    for list_names in [('grc', 'br'), ('eng', 'mr-eng')]:
        lists = [(ocr_vars.DICTIONARIES_DIR / f'{n}.txt').read_text(encoding='utf-8').splitlines()
                 for n in list_names]
        lists = merge_wordlists(*lists)
        new_name = ocr_vars.SEPARATOR.join(list_names)
        (ocr_vars.DICTIONARIES_DIR / f'{new_name}.txt').write_text('\n'.join(lists), encoding='utf-8')
