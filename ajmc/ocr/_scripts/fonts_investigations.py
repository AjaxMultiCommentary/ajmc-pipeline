import json
from pathlib import Path

import pandas as pd
import unicodedata

from ajmc.commons import unicode_utils
from ajmc.ocr.font_utils import Font, walk_through_font_dir
from ajmc.ocr.pytorch.data_generation import draw_textline

#%%

ajmc_charset_path = Path('./data/fonts/glyphs/ajmc_final_glyphs_nfc.txt')
all_ajmc_chars = ajmc_charset_path.read_text(encoding='utf-8')

font_dir = Path('./data/fonts/fonts')

fonts_check = {
    'path': [],
    'name': [],
    'all_ajmc': [],
    'all_ajmc_missing': [],
    'latin': [],
    'latin_missing': [],
    'greek': [],
    'greek_missing': [],
    'modern_greek': [],

}

all_missing = set(all_ajmc_chars)

#%%

for font_path in walk_through_font_dir(font_dir):

    font = Font(font_path, size=100, font_variant='Regular')

    fonts_check['path'].append(font.path)
    fonts_check['name'].append(font.name)

    fonts_check['all_ajmc'].append(font.has_glyphs(all_ajmc_chars))
    fonts_check['all_ajmc_missing'].append(' '.join(sorted(font.get_missing_glyphs(all_ajmc_chars))))

    for charset in ['latin', 'modern_greek', 'greek', ]:
        if charset == 'modern_greek':
            fonts_check[charset].append(font.has_glyphs('ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω'))

        fonts_check[charset].append(font.has_charset_glyphs(charset))

        if charset != 'modern_greek':
            fonts_check[f'{charset}_missing'].append(' '.join(sorted(font.get_missing_glyphs(unicode_utils.CHARSETS_CHARS_NFC[charset]))))

    present = set([char for char in all_ajmc_chars if font.has_glyph(char)])
    all_missing -= present

df = pd.DataFrame.from_dict(fonts_check, orient='columns')

df.to_excel((font_dir.parent / 'glyphs/fonts_glyphs.xlsx'), index=False, encoding='utf-8')

#%% Check missing glyphs manually and graphically
from ajmc.commons import unicode_utils

import shutil

output_dir = Path('/Users/sven/Desktop/fonts_test')
shutil.rmtree(output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

# get all ajmc chars NFC
ALL_AJMC_CHARS_NFD = Path('./data/fonts/glyphs/ajmc_final_glyphs.json').read_text(encoding='utf-8')
ALL_AJMC_CHARS_NFD = ''.join(json.loads(ALL_AJMC_CHARS_NFD).keys())  # This is NFD

ALL_AJMC_CHARS_NFC = {charset: ''.join([c for c in chars if all([c_nfd in ALL_AJMC_CHARS_NFD for c_nfd in unicodedata.normalize('NFD', c)])])
                      for charset, chars in unicode_utils.CHARSETS_CHARS_NFC.items()}

ALL_AJMC_CHARS_NFC = ''.join(ALL_AJMC_CHARS_NFC.values())  # This is NFC

testing_chars = {
    'latin': "123 QWcd À'ÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐ ÑÒÓÔÕÖÙÚÛÜÝß àáâãäåæç èéêëìíîïðñ òóôõö ùúûüŒœ",
    'greek': "Ά·ΈΉΊΌΎΏΐΓΔΖ' άέήίΰαβγδεϚϛ ϜϝϞϟϠϡ ἃἌἍἎἏ ἐἙἠἡἦἧἯἰἱἾἿὀὁὂᾦᾯᾰΆᾼιῂῈῑῒῥῳῴῶΏῼῺ́̓",
    'punctuation': """.,:;!?-–—'"«»()[]{}<>"""
}

FONTS_DIR = Path('./data/fonts/fonts')
FONTS = [Font(font_path, font_variant='Regular') for font_path in walk_through_font_dir(FONTS_DIR)]

for font in FONTS:
    for variant, variant_font in font.font_variants.items():
        fonts = {charset: variant_font for charset in ['latin', 'greek', 'numeral', 'punctuation']}
        fallback_fonts = [variant_font]

        for i in range(0, len(ALL_AJMC_CHARS_NFC), 40):
            draw_textline(ALL_AJMC_CHARS_NFC[i:i + 40], fonts, fallback_fonts=fallback_fonts, target_height=100, kerning=2,
                          output_file=Path(f'/Users/sven/Desktop/fonts_test/{variant_font.path.stem}_{i}.png'))

        # for charset, charset_chars in ALL_AJMC_CHARS_NFC.items():
        #
        #     chars_count = len([char for char in charset_chars if variant_font.has_glyph(char)])
        #     draw_textline(charset_chars, fonts, target_height=100, kerning=2, fallback_fonts=fallback_fonts,
        #                   output_file=Path(f'/Users/sven/Desktop/fonts_test/{variant_font.path.stem}_{charset}_{chars_count}.png'))

#%%
import shutil

text = '479 ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠ Je suis un Lapin. {}'

FONTS_DIR = Path('./data/fonts/fonts')
font_path = FONTS_DIR / 'NotoSans-Regular.ttf'
font = Font(font_path, font_variant='Regular')

fonts = {charset: font for charset in ['latin', 'greek', 'numeral', 'punctuation']}
fonts['fallback'] = [font]

output_dir = Path('/Users/sven/Desktop/degrade_test/')
shutil.rmtree(output_dir)
output_dir.mkdir()

for size in [32, 40, 100]:
    draw_textline(text, fonts, target_height=size, output_file=Path(f'/Users/sven/Desktop/degrade_test/{size}_black.png'))

#%% Create white images
from PIL import Image, ImageOps

for png in output_dir.glob('*.png'):
    img = Image.open(png)
    img = img.convert('L')
    img = ImageOps.invert(img)
    img.save(output_dir / f'{png.stem.split("_")[0]}_white.png')

#%% Test degrade

from ajmc.ocr.pytorch.data_generation import degrade_line, distort_line


for png in output_dir.glob('*.png'):
    img = Image.open(png)
    img = img.convert('L')
    degraded = degrade_line(img)
    degraded.save(output_dir / f'{png.stem}_degr.png')
    distorted = distort_line(img)
    distorted.save(output_dir / f'{png.stem}_dist.png')
