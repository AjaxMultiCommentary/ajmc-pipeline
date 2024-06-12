"""A simple script to generate lines data for the paper."""

from pathlib import Path

from ajmc.commons import variables as vs
from ajmc.ocr.data_processing.data_generation import draw_textline
from ajmc.ocr.data_processing.font_utils import Font, get_fallback_fonts


# Generate the wecklein1894_0108_line_example
text = 'Nauck als unecht bezeichnet.  330 λόγῳ f. φίλοι Stob. flor. 113, 8.'

fonts = {'latin': Font(vs.FONTS_DIR / 'ModerneFraktur-Regular.ttf'),
         'greek': Font(vs.FONTS_DIR / 'Goschen-Regular.otf'),
         'numeral': Font(vs.FONTS_DIR / 'BodoniFLF-Regular.ttf'),
         'punctuation': Font(vs.FONTS_DIR / 'Didot-Regular.otf')}

output_path = Path('/scratch/sven/Wecklein1894_example_lines/Wecklein1894_0108_line_example_generated_fraktur_goschen.png')

fallback_fonts = get_fallback_fonts()

draw_textline(text, fonts, fallback_fonts, 80,
              font_variants=['Regular'] * 30 + ['Bold'] * 4 + ['Regular'] * 34,
              kerning=1,
              output_file=output_path,
              )

# Generate a baskerville porson line
fonts = {'latin': Font(vs.FONTS_DIR / 'LibreBaskerville-Regular.ttf'),
         'greek': Font(vs.FONTS_DIR / 'Porson-Regular.otf'),
         'numeral': Font(vs.FONTS_DIR / 'Caladea-Regular.ttf'),
         'punctuation': Font(vs.FONTS_DIR / 'LibreBaskerville-Regular.ttf')}

output_path = Path('/scratch/sven/Wecklein1894_example_lines/Wecklein1894_0108_line_example_generated_Baskerville_Porson.png')

draw_textline(text, fonts, fallback_fonts, 80,
              font_variants=['Italic'] * 6 + ['Regular'] * 24 + ['Bold'] * 9 + ['Regular'] * 28,
              kerning=1,
              output_file=output_path,
              )
