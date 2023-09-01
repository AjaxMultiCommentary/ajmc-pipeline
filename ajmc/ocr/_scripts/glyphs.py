"""The goal of this script is to compare the glyphs present in the corpora with the glyphs present in the GF_glyphsets and to cherry-pick
a definitve glyphset for the ocr.

Hint: By default, a glyphset is a string object."""

import json
import unicodedata
from ajmc.corpora.corpora_classes import Corpus
from pathlib import Path
import re
from typing import Dict, List, Tuple, Union


def get_glyph_name(glyph: str) -> str:
    try:
        return unicodedata.name(glyph)
    except ValueError:
        return 'NO UNICODE NAME'


def get_glyphs_name_table(glyphs: str) -> List[Tuple[str, str]]:
    return [(g, get_glyph_name(g)) for g in glyphs]


def write_glyphsets_name_table(glyph_name_table: Dict[str, List[Tuple[str, str]]], output_path: Union[str, Path]) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        for set_name, glyph_name_table in glyph_name_table.items():
            f.write(f'{set_name}\n')
            for glyph, name in glyph_name_table:
                f.write(f'{glyph}\t{name}\n')
            f.write('\n\n')


def get_nfd_glyphset(glyphset: str) -> str:
    nfd_string = ''.join([unicodedata.normalize('NFD', g) for g in glyphset])
    return ''.join(sorted(set(nfd_string)))


#Compare with the chars present in the corpus
corpus_ids = [
    'First1KGreek',
    'corpus_thomisticum',
    'logeion',
    'agoraclass',
    'corpus_scriptorum_latinorum',
    'forum_romanum',
    'mediterranee_antique',
    'the_latin_library',
    'perseus_secondary',
    'perseus_legacy'
]

glyphs_dir = Path('./data/fonts/glyphs')
#%% Start with google fonts
gf_glyphsets_path = glyphs_dir / 'gf_glyphsets.json'
gf_glyphsets = json.loads(gf_glyphsets_path.read_text(encoding='utf-8'))

# write gf_glyphsets_nfd
gf_glyphsets_nfd = {k: get_nfd_glyphset(v) for k, v in gf_glyphsets.items()}
gf_glyphsets_path.with_name('gf_glyphsets_nfd.json').write_text(json.dumps(gf_glyphsets_nfd, indent=4, ensure_ascii=False), encoding='utf-8')

# write gf_glyphsets_nfd_table
gf_glyphsets_nfd_table = {k: get_glyphs_name_table(v) for k, v in gf_glyphsets_nfd.items()}
write_glyphsets_name_table(gf_glyphsets_nfd_table, gf_glyphsets_path.with_name('gf_glyphsets_nfd_table.txt'))

#%% Create gf_ajmc_glyphsets (a subset of gf_glyphsets)

ajmc_glyphsets = ["GF_Greek_Core",
                  "GF_Greek_Plus",
                  "GF_Greek_Pro",
                  "GF_Latin_Beyond",
                  "GF_Latin_Core",
                  "GF_Latin_Kernel",
                  "GF_Latin_Plus"]

gf_ajmc_glyphsets = {k: ''.join(sorted(v)) for k, v in gf_glyphsets.items() if k in ajmc_glyphsets}
gf_glyphsets_path.with_name('gf_ajmc_glyphsets.json').write_text(json.dumps(gf_ajmc_glyphsets, indent=4, ensure_ascii=False), encoding='utf-8')

# write gf_ajmc_glyphsets_nfd
gf_ajmc_glyphsets_nfd = {k: get_nfd_glyphset(v) for k, v in gf_ajmc_glyphsets.items()}
gf_glyphsets_path.with_name('gf_ajmc_glyphsets_nfd.json').write_text(json.dumps(gf_ajmc_glyphsets_nfd, indent=4, ensure_ascii=False),
                                                                     encoding='utf-8')

# write gf_ajmc_glyphsets_nfd_table
gf_ajmc_glyphsets_nfd_table = {k: get_glyphs_name_table(v) for k, v in gf_ajmc_glyphsets_nfd.items()}
write_glyphsets_name_table(gf_ajmc_glyphsets_nfd_table, gf_glyphsets_path.with_name('gf_ajmc_glyphsets_nfd_table.txt'))

#%% Get the set of glyphs present in the corpora

corpora_glyph_counter_path = glyphs_dir / 'corpora_glyphs_counts.json'
corpora_glyph_counter = json.loads(corpora_glyph_counter_path.read_text(encoding='utf-8'))

# Get the set of glyphs present in the corpora
corpora_glyphset = ''.join([k for k in corpora_glyph_counter.keys() if re.sub(r'\s+', '', k) != ''])
(glyphs_dir / 'corpora_glyphset.json').write_text(json.dumps(corpora_glyphset, indent=4, ensure_ascii=False), encoding='utf-8')

# Get the set of glyphs present in the corpora in NFD
corpora_glyphset_nfd = get_nfd_glyphset(corpora_glyphset)
(glyphs_dir / 'corpora_glyphset_nfd.json').write_text(json.dumps(corpora_glyphset_nfd, indent=4, ensure_ascii=False), encoding='utf-8')

#Set a threshold for the number of times a glyph must appear in the corpora

thresholded_corpora_glyphs = set([g for g, c in corpora_glyph_counter.items() if c > 50 and re.sub(r'\s+', '', g) != ''])

#%% Get the set of glyphs present in GT4HistOCR
gt4_glyphs_path = glyphs_dir / 'gt4_glyphs.json'
gt4_glyphs = json.loads(gt4_glyphs_path.read_text(encoding='utf-8'))

# Get the nfd
gt4_glyphs_nfd = get_nfd_glyphset(gt4_glyphs)
gt4_glyphs_path.with_name('gt4_glyphs_nfd.json').write_text(json.dumps(gt4_glyphs_nfd, indent=4, ensure_ascii=False), encoding='utf-8')

# Get the table
gt4_glyphs_nfd_table = get_glyphs_name_table(gt4_glyphs_nfd)
write_glyphsets_name_table({'gt4': gt4_glyphs_nfd_table}, gt4_glyphs_path.with_name('gt4_glyphs_nfd_table.txt'))

#%% Create a manual table to cherry pick the glyphs

table = 'GLYPH\tNAME\tCORPORA\tGT4\n'

all_glyphs = set()

for glyphset_name, glyphset in gf_ajmc_glyphsets_nfd.items():
    table += f'# {glyphset_name}\n'
    for glyph in glyphset:
        if glyph not in all_glyphs:
            table += f'{glyph}\t{get_glyph_name(glyph)}\t{corpora_glyph_counter.get(glyph, 0)}\t{glyph in gt4_glyphs_nfd}\n'
            all_glyphs.add(glyph)
    table += '\n\n'

table += f'# Corpora\n'
for glyph in corpora_glyphset_nfd:
    if not any([glyph in gs for gs in gf_ajmc_glyphsets_nfd.values()]):
        if glyph not in all_glyphs:
            table += f'{glyph}\t{get_glyph_name(glyph)}\t{corpora_glyph_counter.get(glyph, 0)}\t{glyph in gt4_glyphs_nfd}\n'
            all_glyphs.add(glyph)
table += '\n\n'

table += f'# GT4\n'
for glyph in gt4_glyphs_nfd:
    if not any([glyph in gs for gs in gf_ajmc_glyphsets_nfd.values()]):
        table += f'{glyph}\t{get_glyph_name(glyph)}\t{corpora_glyph_counter.get(glyph, 0)}\t{True}\n'

# (glyphs_dir / 'glyphs_manual_table.tsv').write_text(table, encoding='utf-8')


#%% We are now ready to compare the corpora glyphs with the gf_ajmc_glyphsets
for k, v in gf_ajmc_glyphsets_nfd.items():
    print(k)
    for g in v:
        if g not in corpora_glyphset_nfd:
            print(g, get_glyph_name(g))

    if input('Press enter to continue') != '':
        break

#%% Search for a character in the corpus.

corpus = Corpus.auto_init('logeion')
text = corpus.get_plain_text()

corpus_perseus = Corpus.auto_init('perseus_secondary')
text_perseus = corpus_perseus.get_plain_text()

text_first_k = Corpus.auto_init('First1KGreek').get_plain_text()


def search_char(char, text, window):
    printed = 0
    for i in range(len(text)):
        if printed > 10:
            break
        if text[i] == char:
            print(text[i - window:i + window + 1])
            printed += 1


#%% blabla

for text_ in [text, text_perseus, text_first_k]:
    search_char('‡', text_, 20)

#%% Read manually created table and export it

glyphs_manual_table_path = glyphs_dir / 'glyphs_manual_table.tsv'
glyphs_manual_table = glyphs_manual_table_path.read_text(encoding='utf-8')

ajmc_final_glyphs = {}
for line in glyphs_manual_table.split('\n'):
    if line:
        if line[0] != '#':
            glyph, name, corpora, gt4 = line.split('\t')
            ajmc_final_glyphs[glyph] = name

ajmc_final_glyphs_path = glyphs_dir / 'ajmc_final_glyphs.json'
ajmc_final_glyphs_path.write_text(json.dumps(ajmc_final_glyphs, indent=4, ensure_ascii=False), encoding='utf-8')

ajmc_glyphs_nfd = set(get_nfd_glyphset(''.join(ajmc_final_glyphs.keys())))

#%% Check missing chars from PoGreTra
pog_dir = Path('/scratch/sven/ocr_exp/datasets/pog')
pog_glyphs = set(''.join([l.read_text(encoding='utf-8') for l in pog_dir.glob('*.txt')]))
pog_glyphs_nfd = set(''.join([unicodedata.normalize('NFD', g) for g in pog_glyphs]))
pog_glyphs_nfd - ajmc_glyphs_nfd

#%% Check missing chars from GT4
from tqdm import tqdm

gt4_dir = Path('/scratch/sven/ocr_exp/source_datasets/gt4histocr')
gt4_glyphs = set(''.join([l.read_text(encoding='utf-8') for l in tqdm(gt4_dir.rglob('*.txt'))]))
gt4_glyphs_nfd = set(''.join([unicodedata.normalize('NFD', g) for g in gt4_glyphs]))

Path('data/fonts/glyphs/gt4_glyphs_nfc.txt').write_text('\n'.join(sorted(gt4_glyphs_nfd)), encoding='utf-8')
#%%
for char in gt4_glyphs_nfd - ajmc_glyphs_nfd:
    try:
        name = unicodedata.name(char)
    except ValueError:
        name = 'NO UNICODE NAME'
    print(char, '\t', name)

#%% Write the results
Path('./data/fonts/glyphs/ajmc_glyphs_nfd_table.json').write_text(json.dumps(ajmc_glyphs_nfd_table, indent=4, sort_keys=True), encoding='utf-8')
#%% GET GLYPHS FROM FONTS (TO TABLE)


# get the NFD form of the character
print(unicodedata.normalize('NFD', 'ἀ'))
a = unicodedata.normalize('NFD', 'ᾳ')
a.encode('utf-8')
'ἀ'.encode('utf-8')
[unicodedata.name(c) for c in a]

from pathlib import Path
import json

font_dir = Path('/Users/sven/packages/ajmc/data/fonts/fonts')
greek_font_dir = font_dir / 'greek_fonts'
latin_font_dir = font_dir / 'latin_fonts'

gf_glyphsets = json.loads(Path('/Users/sven/packages/ajmc/data/fonts/glyphsets.json').read_text(encoding='utf-8'))

#%% Invesigate glyphsets

import json
from pathlib import Path
import unicodedata

path = Path('/Users/sven/packages/ajmc/data/fonts/AJMC_glyphsets.json')
ajmc_glyphsets = json.loads(path.read_text(encoding='utf-8'))

# All glyphs
print('all glyphs:    ', sum([len(gs) for gs in ajmc_glyphsets.values()]))

# All unique glyphs
unique_glyphs = set()
for glyphs in ajmc_glyphsets.values():
    for glyph in glyphs:
        unique_glyphs.add(glyph)
print('all unique glyphs:  ', len(unique_glyphs))

# All unique glyphs with normal form Decomposed
unique_glyphs_decomposed = set()
for glyphs in ajmc_glyphsets.values():
    for glyph in glyphs:
        for subglyph in unicodedata.normalize('NFD', glyph):
            unique_glyphs_decomposed.add(subglyph)

print('all unique glyphs decomposed:  ', len(unique_glyphs_decomposed))

#%% PENDING functions


corpora_glyph_counter['a']


def combined_to_standalone_unicode(text: str) -> str:
    """Converts combined unicode characters to standalone unicode characters.

    Args:
        text (str): The text to convert, in NFD form

    Returns:
        str: The converted text.
    """
    for match in regex.findall(r'\p{M} ', text):
        char_name = get_glyph_name(match[0])
        if char_name.startswith('COMBINING '):
            replace_char = unicodedata.lookup(char_name.replace('COMBINING ', ''))
            text = text.replace(match, replace_char + ' ')

    return text


#%%

ajmc_glyphset_nfc = set([glyph for charset in gf_glyphsets.values() for glyph in charset
                         if all([c in ajmc_final_glyphs for c in unicodedata.normalize('NFD', glyph)])])

Path('./data/fonts/glyphs/ajmc_final_glyphs_nfc.txt').write_text(''.join(sorted(ajmc_glyphset_nfc)), encoding='utf-8')
