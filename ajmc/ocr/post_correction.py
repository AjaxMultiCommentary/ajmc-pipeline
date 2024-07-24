"""This module contains functions for post-correction of OCR'd text."""

import re

import langid
import Levenshtein
from spellchecker import SpellChecker
from tqdm import tqdm

from ajmc.commons import unicode_utils as uu
from ajmc.ocr import variables as ocr_vs


def create_post_correction_dictionaries():
    ocr_vs.POST_CORRECTION_DICTIONARIES_DIR.mkdir(parents=True, exist_ok=True)

    # English
    to_merge = [('en', 'eng', 'mr-eng'),
                ('de', 'deu', 'mr-ger', 'frk'),
                ('fr', 'mr-fre'),
                ('ita', 'mr-ita'),
                ('grc', 'br', 'mr-gre'),
                ('lat', 'mr-lat'),
                ]

    for lang_names in to_merge:
        words = set()
        for lang_name in lang_names:
            words.update((ocr_vs.DICTIONARIES_DIR / f'{lang_name}.txt').read_text(encoding='utf-8').splitlines())

        # Clean the dictionary
        # We start by trimming each word of punctuation signs at the beginning and end
        punctuation = uu.get_all_chars_from_ranges(uu.CHARSETS_RANGES['punctuation'])
        words = set(word.strip(punctuation) for word in words)

        # We remove words containing numbers
        words = set(word for word in words if not any(char.isdigit() for char in word))

        # We capitalize the first letter of each word to speed things up (we are not looking for perfection and models rarely make mistakes with caps)
        words = set(word.capitalize() for word in words)

        # Write the dictionary
        (ocr_vs.POST_CORRECTION_DICTIONARIES_DIR / f"{lang_names[0]}.txt").write_text('\n'.join(words), encoding='utf-8')


def rebuild_word_from_candidate(left_punctuation: str, core_word: str, right_punctuation: str, candidate: str) -> str:
    """Rebuild a word."""
    is_capitalized = core_word[0].isupper()
    is_all_caps = core_word.isupper()

    if is_all_caps:
        return left_punctuation + candidate.upper() + right_punctuation
    elif is_capitalized:
        return left_punctuation + candidate + right_punctuation
    else:
        return left_punctuation + candidate.lower() + right_punctuation


def correct_word(word: str, word_set: set, threshold: int, max_candidates: int = 1,
                 method: str = 'Dict. only', spellcheck_function=None) -> str:
    """Correct a word."""
    if "'" in word:
        return word

    # Save punctuation signs right and left of the word using regex
    left_punctuation = re.match(r'^(\W*)', word).group(1)
    right_punctuation = re.match(r'^(\W*)', word[::-1]).group(1)[::-1]  # reversing as $-based regex seem to be buggy in python
    core_word = word[len(left_punctuation):len(word) - len(right_punctuation)]
    if not core_word:
        return word

    if method == 'Dict. only':
        # Capitalize  # For titles
        capitalized = core_word.capitalize()

        if capitalized in word_set:
            return word
        else:
            candidates = []
            for w in word_set:
                if Levenshtein.distance(w, capitalized) <= threshold:
                    candidates.append((w, Levenshtein.editops(capitalized, w)[0][0]))
                if len(candidates) > max_candidates:
                    return word

            if 0 > len(candidates) <= max_candidates:
                return rebuild_word_from_candidate(left_punctuation, core_word, right_punctuation, candidates[0][0])

            else:
                return word
    elif method == 'pyspellchecker':
        corrected = spellcheck_function(core_word)
        return rebuild_word_from_candidate(left_punctuation, core_word, right_punctuation, core_word if corrected is None else corrected)


word_sets = {p.stem[:2]: set(p.read_text(encoding='utf-8').splitlines()) for p in ocr_vs.POST_CORRECTION_DICTIONARIES_DIR.glob('*.txt')}


def correct_string(string_to_correct: str, lev_max_distance: int = 1, max_candidates: int = 1,
                   method: str = 'Dict. only') -> str:
    """Correct a string."""

    words = string_to_correct.split()

    # We get the charset of each word
    labels = [uu.get_string_charset(word) for word in words]

    # We automatically detect the language of the text
    label = langid.classify(' '.join(w for w, c in zip(words, labels) if c == 'latin'))[0]

    if label not in ['en', 'de', 'fr'] and method != 'Dict. only':
        return string_to_correct

    # We reassign the language based on the charset
    charsets = [label if charset == 'latin' else 'gr' if charset == 'greek' else charset for charset in labels]

    if method == 'pyspellchecker':
        spell = SpellChecker(language=label)

    corrected_text = []
    for word, charset in zip(words, charsets):
        if charset in ['punctuation', 'numeral'] or charset not in word_sets:
            corrected_text.append(word)
            continue
        if charset == 'gr':
            corrected_text.append(correct_word(word, word_sets['gr'], lev_max_distance, max_candidates=max_candidates, method='Dict. only'))
        else:
            corrected_text.append(correct_word(word, word_sets[charset], lev_max_distance, max_candidates=max_candidates, method=method,
                                               spellcheck_function=spell.correction if method == 'pyspellchecker' else None))
    return ' '.join(corrected_text)


# Read jsonl files to get the data
data_dir = ocr_vs.EXPERIMENTS_DIR.parent / 'post_ocr_correction'
import json

data = {}
for key in ['ajmc_primary_text', 'ajmc_mixed']:
    data[key] = (data_dir / f'{key}.jsonl').read_text(encoding='utf-8').splitlines()
    data[key] = [json.loads(line) for line in data[key]]
    data[key] = [(d['ocr']['line'], d['groundtruth']['line']) for d in data[key]]
    # data[key] = random.sample(data[key], k=min(200, len(data[key]))) # For testing purposes

#%%
for ocr, gt in data['ajmc_mixed']:
    if ocr != gt:
        print(ocr)
        print(correct_string(ocr, method='pyspellchecker'))
        print(gt)
        print()

#%%
possible_improvements = {key: [] for key in data}
for key in data:
    for ocr, gt in tqdm(data[key]):
        possible_improvements[key].append(Levenshtein.distance(ocr, gt) / len(gt))

# mean improvement

possible_improvements = {key: sum(possible_improvements[key]) / len(possible_improvements[key]) for key in possible_improvements}

#%%
improvements = {key: [] for key in data}
index = []
for method in ['Dict. only', 'Dict. only (10)', 'pyspellchecker']:
    for key in data:
        current_improvements = []
        for ocr, gt in tqdm(data[key]):
            uncorrected_distance = Levenshtein.distance(ocr, gt)
            if method == 'Dict. only (10)':
                corrected_ocr = correct_string(ocr, method='Dict. only', max_candidates=10)
            else:
                corrected_ocr = correct_string(ocr, method=method)
            corrected_distance = Levenshtein.distance(corrected_ocr, gt)
            current_improvements.append((uncorrected_distance - corrected_distance) / len(gt))

        # mean improvement
        improvements[key].append(sum(current_improvements) / len(current_improvements))

    index.append(('Dictionary', method))

models = ['LLAMA-7B', 'LLAMA-2-7B', 'BLOOM-560M', 'BLOOM-3B', 'BLOOM-7.1B', 'BLOOMZ-560M', 'BLOOMZ-3B', 'BLOOMZ-7.1B', 'OPT-350', 'OPT-6.7B', 'GPT-2',
          'GPT-3', 'GPT-3.5', 'GPT-4']
results = {'ajmc_mixed': [-0.21, -0.15, -0.63, -0.29, -0.22, -0.72, -0.64, -0.16, -0.78, -0.34, -0.76, -0.74, -0.74, -0.37],
           'ajmc_primary_text': [-0.49, -0.22, -0.63, -0.56, -0.73, -0.70, -0.69, -0.51, -0.78, -0.67, -0.60, -0.43, -0.08, 0.06]}

index += [('Generative', model) for model in models]
for i, model in enumerate(models):
    for key in results:
        improvements[key].append(results[key][i])

# We now create a dataframe
import pandas as pd

df = pd.DataFrame.from_dict(improvements)
df.index = pd.MultiIndex.from_tuples(index, names=['Type', 'Method'])

# Rename the columns with a dict of names
df = df.rename(columns={'ajmc_primary_text': 'Greek only', 'ajmc_mixed': 'Mixed'})

df['All'] = df.mean(axis=1)

styler = df.style

# Bold the best value in each column
styler = styler.apply(lambda x: ['font-weight: bold' if v == x.max() else '' for v in x], axis=0)

# Export this dataframe to latex
styler.format(escape='latex', precision=2)

# Add a midrule between the two types
styler.set_caption(
    """Post OCR correction results. The values correspond to the mean normalized improvement in the Levenshtein distance. Values of the generative group are taken from the original paper.""")
styler.set_table_styles()

latex = styler.to_latex(hrules=True,
                        position_float='centering',
                        convert_css=True,
                        label=f'tab:4_1 {styler.caption.split(".")[0]}')

#%%
# We now test the function
# from pathlib import Path
# import pandas as pd
# import unicodedata
#
# # Load the data
# data = pd.read_csv(Path('/scratch/sven/withbackbone_v2/eval_1525000/error_record.tsv'), sep='\t', encoding='utf-8')
#
# improvements = []
# for ocr, gt in tqdm(zip(data['ocr'][:100], data['gt'][:100])):
#     ocr = unicodedata.normalize('NFC', ocr)
#     gt = unicodedata.normalize('NFC', gt)
#     uncorrected_distance = Levenshtein.distance(ocr, gt)
#
#     # if uncorrected_distance <= 4:
#     #     continue
#     # else:
#     corrected_ocr = correct_string(ocr, method='pyspellchecker')
#     corrected_distance = Levenshtein.distance(corrected_ocr, gt)
#     improvements.append(uncorrected_distance - corrected_distance)
#     if ocr != corrected_ocr:
#         print()
#         print(gt)
#         print(ocr)
#         print(corrected_ocr)
#         print()

#%%
