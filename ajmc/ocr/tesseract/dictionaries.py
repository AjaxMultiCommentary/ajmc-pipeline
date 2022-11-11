"""Utils to handle tesseract dictionaries, BR and MR's dictionaries and combine them with tesseract's dictionaries
to create new models."""

import os
from pathlib import Path
from typing import List
from ajmc.commons.miscellaneous import get_custom_logger
from ajmc.commons.variables import TESS_XP_DIR
import pandas as pd

from ajmc.ocr.utils import is_greek_string, is_latin_string, is_number_string

logger = get_custom_logger(__name__)


def get_trainneddata_path(traineddata_name: str) -> Path:
    """Get the path to a traineddata file"""
    return TESS_XP_DIR / 'traineddata' / (traineddata_name + '.traineddata')


def get_traineddata_unpacked_dir(traineddata_name: str) -> Path:
    """Get the path to the unpacked directory of a traineddata file"""
    return TESS_XP_DIR / 'traineddata_unpacked' / traineddata_name


def write_unpacked_traineddata(traineddata_path: Path,
                               unpacked_dir: Path = None):
    """Unpacks a traineddata file"""

    # Set path to custom data
    if unpacked_dir is None:
        unpacked_dir = get_traineddata_unpacked_dir(traineddata_path.stem)

    unpacked_dir.mkdir(parents=True, exist_ok=True)

    # Unpack traineddata
    command = f"combine_tessdata -u {traineddata_path} {unpacked_dir}/{traineddata_path.stem}."
    os.system(command)


def get_traineddata_wordlist(traineddata_name) -> List[str]:
    """Get the wordlist of a traineddata file"""

    write_unpacked_traineddata(get_trainneddata_path(traineddata_name))
    unpacked_dir = get_traineddata_unpacked_dir(traineddata_name)
    # Get wordlist
    dawg_path = unpacked_dir / f'{traineddata_name}.lstm-word-dawg'
    unicharset_path = unpacked_dir / f'{traineddata_name}.lstm-unicharset'
    wordlist_path = TESS_XP_DIR / 'utils/dictionaries' / f'{traineddata_name}.txt'
    command = f'dawg2wordlist {unicharset_path} {dawg_path} {wordlist_path}'
    os.system(command)

    return wordlist_path.read_text(encoding='utf-8').splitlines()


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
    dfa = pd.read_csv(TESS_XP_DIR / 'utils/dictionaries' / 'mr_authors.tsv', encoding='utf-8', sep='\t',
                      keep_default_na=False)
    dfw = pd.read_csv(TESS_XP_DIR / 'utils/dictionaries' / 'mr_works.tsv', encoding='utf-8', sep='\t',
                      keep_default_na=False)
    df = pd.concat([dfa, dfw], axis=0)

    # Define a filter
    def filter_func(x):
        if "gre" in langs:
            return x['TYPE'] in langs and is_greek_string(x['TEXT'], threshold=0.9)
        else:
            return x['TYPE'] in langs and is_latin_string(x['TEXT'], threshold=0.9)

    # Creates the Filter
    filter = df.apply(filter_func, axis=1)

    # Apply the filter
    abbreviations = df[filter]['TEXT'].tolist()

    # One word per line
    abbreviations = list(set(' '.join(abbreviations).split()))

    # Delete numbers
    abbreviations = [x for x in abbreviations if not is_number_string(x, threshold=0.9)]

    return abbreviations


def write_mr_abbr_wordlist(langs: List[str]):
    """Write the wordlist of the MR abbreviation dictionary"""
    wordlist = get_mr_abbr_wordlist(langs)
    file_name = f'mr-{langs[0]}.txt'
    (TESS_XP_DIR / 'utils/dictionaries' / file_name).write_text('\n'.join(wordlist), encoding='utf-8')


def change_traineddata_wordlist(traineddata_name, wordlist_name):
    """Change the wordlist of a traineddata file"""

    write_unpacked_traineddata(get_trainneddata_path(traineddata_name))

    # Get the path to the unpacked directory
    unpacked_dir = get_traineddata_unpacked_dir(traineddata_name)
    wordlist_path = TESS_XP_DIR / 'utils/dictionaries' / wordlist_name + '.txt'

    # Copy the wordlist
    command = f'wordlist2dawg {wordlist_path} {unpacked_dir}/{traineddata_name}.lstm-unicharset {unpacked_dir}/{traineddata_name}.lstm-word-dawg'
    os.system(command)

    # Pack the traineddata
    command = f'combine_tessdata {unpacked_dir}/{traineddata_name}.'
    os.system(command)

    # move the traineddata
    command = f'mv {unpacked_dir}/{traineddata_name}.traineddata {TESS_XP_DIR}/traineddata/{traineddata_name}_{wordlist_name}.traineddata'
    os.system(command)

    # Delete the unpacked directory
    command = f'rm -r {unpacked_dir}'
    os.system(command)



def write_all_wordlists():
    """Write all wordlists"""
    write_mr_abbr_wordlist(['gre'])
    write_mr_abbr_wordlist(['lat'])
    write_mr_abbr_wordlist(['eng'])
    write_mr_abbr_wordlist(['ita'])
    write_mr_abbr_wordlist(['fre'])
    write_mr_abbr_wordlist(['ger'])
    write_mr_abbr_wordlist(['spa'])

    for path in TESS_XP_DIR.glob('traineddata/*.traineddata'):
        wordlist = get_traineddata_wordlist(path.stem)
        file_name = f'{path.stem}.txt'
        (TESS_XP_DIR / 'utils/dictionaries' / file_name).write_text('\n'.join(wordlist), encoding='utf-8')



def create_trained_data_with_dicts(traineddata_name,
                                   wordlist_names: List[str],):

    wordlists = [(TESS_XP_DIR/'utils/dictionaries'/f'{wordlist_name}.txt').read_text('utf-8').splitlines()
                 for wordlist_name in wordlist_names]

    # merge the two list
    merged_wordlist = merge_wordlists(*wordlists)

    # write the new wordlist
    merged_wordlist_name = '+'.join(wordlist_names)
    (TESS_XP_DIR / 'utils/dictionaries' / f'{merged_wordlist_name}.txt').write_text('\n'.join(merged_wordlist), encoding='utf-8')

    # Change the traineddata's wordlist
    change_traineddata_wordlist(traineddata_name=traineddata_name,
                                wordlist_name=merged_wordlist_name)

