from pathlib import Path

import ajmc.ocr.pytorch.data_generation as dg
from ajmc.ocr.font_utils import Font
from ajmc.ocr.pytorch.data_generation import draw_textline


# Todo : actualise this test
def test_draw_textline():
    strings = [  #'pluribus ue ſi placuiſſ& conſtituiſyllabas facit: quæ ad coniũgendas demum ſubiectas ſibi uocales',
        '180. γπαραλλαγή λέξεων — From Dindorf we read hello I love you.',
        'Coucou']

    fonts_path = Path('/Users/sven/packages/ajmc/data/fonts/fonts')
    fonts = [Font((fonts_path / 'Cardo-Regular.ttf'))]

    fonts_dict = {charset: fonts[0] for charset in ['latin', 'greek', 'numeral', 'punctuation']}
    fallback_fonts = [fonts[0]]

    draw_textline(textline=strings[0], fonts=fonts_dict, fallback_fonts=fallback_fonts, target_height=48,
                  output_file=Path('/Users/sven/Desktop/test.png'),
                  font_variants=['Regular'] * len(strings[0]))


def get_textline_font_variants():
    textline = 'The quick brown fox jumps over the lazy dog.'

    dg.get_textline_font_variants(textline)
    assert len(dg.get_textline_font_variants(textline)) == len(textline)
    assert all([v == 'Regular' for i, v in enumerate(dg.get_textline_font_variants(textline)) if textline[i] == ' '])
