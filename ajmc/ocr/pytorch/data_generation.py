"""This scripts contains function for generating synthetic data. ``pil2array``, ``array2pil``,
``ocropy_degrade``, ``degrade_line`` and ``distort_line`` are directly copied from kraken to avoid
import issues. """

import argparse
import json
import random
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Generator

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter, distance_transform_cdt, binary_closing, affine_transform, geometric_transform
from tqdm import tqdm

from ajmc.commons import unicode_utils
from ajmc.commons.file_management import int_to_x_based_code
from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.commons.unicode_utils import get_char_charset
from ajmc.corpora.corpora_classes import Corpus
from ajmc.ocr.font_utils import Font, walk_through_font_dir

logger = get_ajmc_logger(__name__)

def pil2array(im: Image.Image, alpha: int = 0) -> np.ndarray:
    """Kraken linegen code."""
    if im.mode == '1':
        return np.array(im.convert('L'))
    return np.array(im)


def array2pil(a: np.ndarray) -> Image.Image:
    if a.dtype == np.dtype("B"):
        if a.ndim == 2:
            return Image.frombytes("L", (a.shape[1], a.shape[0]),
                                   a.tobytes())
        elif a.ndim == 3:
            return Image.frombytes("RGB", (a.shape[1], a.shape[0]),
                                   a.tobytes())
        else:
            raise Exception("bad image rank")
    elif a.dtype == np.dtype('float32'):
        return Image.frombytes("F", (a.shape[1], a.shape[0]), a.tobytes())
    else:
        raise Exception("unknown image type")


def degrade_line(im, eta=0.0, alpha=1.5, beta=1.5, alpha_0=1.0, beta_0=1.0):
    """
    Degrades a line image by adding noise.

    For parameter meanings consult "A statistical, nonparametric methodology for document degradation model validation" (2000) by Tanugo et al.

    Args:
        im (PIL.Image): Input image
        eta (float): Between 0 and 1. And jitters to image. Recommend staying to max 0.05
        alpha (float): Seems to be the concentration of the noise around letters. The higher the more concentrated.Go between 0.1 and 3.
        beta (float): the amount of holes in letters. Try a few 0.1, 0.3 and default.
        alpha_0 (float): No clue what these do, leave default or see paper.
        beta_0 (float): No clue what these do, leave default or see paper.

    Returns:
        PIL.Image in mode '1'
    """
    logger.debug('Inverting and normalizing input image')
    im = pil2array(im)
    im = np.amax(im) - im
    im = im * 1.0 / np.amax(im)

    logger.debug('Calculating foreground distance transform')
    fg_dist = distance_transform_cdt(1 - im, metric='taxicab')
    logger.debug('Calculating flip to white probability')
    fg_prob = alpha_0 * np.exp(-alpha * (fg_dist ** 2)) + eta
    fg_prob[im == 1] = 0
    fg_flip = np.random.binomial(1, fg_prob)

    logger.debug('Calculating background distance transform')
    bg_dist = distance_transform_cdt(im, metric='taxicab')
    logger.debug('Calculating flip to black probability')
    bg_prob = beta_0 * np.exp(-beta * (bg_dist ** 2)) + eta
    bg_prob[im == 0] = 0
    bg_flip = np.random.binomial(1, bg_prob)

    # flip
    logger.debug('Flipping')
    im -= bg_flip
    im += fg_flip

    logger.debug('Binary closing')
    sel = np.array([[1, 1], [1, 1]])
    im = binary_closing(im, sel)
    logger.debug('Converting to image')
    return array2pil(255 - im.astype('B') * 255)


def distort_line(im, distort=3.0, sigma=10, eps=0.03, delta=0.3):
    """
    Distorts a line image.

    Run BEFORE degrade_line as a white border of 5 pixels will be added.

    Args:
        im (PIL.Image): Input image
        distort (float): Distorting of the image set between 1.5 and 4.0, with majority around 2.5
        sigma (float): distorting of the strokes, set between 0.5, 1.5 for 5% of image, else default
        eps (float): set default to 80%, else random between 0.01 and 0.1
        delta (float):

    Returns:
        PIL.Image in mode 'L'
    """
    w, h = im.size
    # XXX: determine correct output shape from transformation matrices instead
    # of guesstimating.
    logger.debug('Pasting source image into canvas')
    image = Image.new('L', (int(1.5 * w), 4 * h), 255)
    image.paste(im, (int((image.size[0] - w) / 2), int((image.size[1] - h) / 2)))
    line = pil2array(image.convert('L'))

    # shear in y direction with factor eps * randn(), scaling with 1 + eps *
    # randn() in x/y axis (all offset at d)
    logger.debug('Performing affine transformation')
    m = np.array([[1 + eps * np.random.randn(), 0.0], [eps * np.random.randn(), 1.0 + eps * np.random.randn()]])
    c = np.array([w / 2.0, h / 2])
    d = c - np.dot(m, c) + np.array([np.random.randn() * delta, np.random.randn() * delta])
    line = affine_transform(line, m, offset=d, order=1, mode='constant', cval=255)

    hs = gaussian_filter(np.random.randn(4 * h, int(1.5 * w)), sigma)
    ws = gaussian_filter(np.random.randn(4 * h, int(1.5 * w)), sigma)
    hs *= distort / np.amax(hs)
    ws *= distort / np.amax(ws)

    def _f(p):
        return (p[0] + hs[p[0], p[1]], p[1] + ws[p[0], p[1]])

    logger.debug('Performing geometric transformation')
    im = array2pil(geometric_transform(line, _f, order=1, mode='nearest'))
    logger.debug('Cropping canvas to content box')
    im = im.crop(ImageOps.invert(im).getbbox())
    return im


def draw_textline(textline: str,
                  fonts: Dict[str, Font],
                  fallback_fonts: List[Font],
                  target_height: int,
                  font_variants: List[str],
                  kerning: int = 0,
                  default_charset: str = 'latin',
                  output_file: Optional[Path] = None,
                  show_image: bool = False) -> Image:
    """
    Draws a textline using the given fonts.

    Args:
        textline (str): The text to draw.
        fonts (dict): A dictionary of fonts, keyed by charset:
            `{'latin': ajmc.ocr.font_utils.Font,
            'greek': ajmc.ocr.font_utils.Font,
            'numeral': ajmc.ocr.font_utils.Font,
            'punctuation': ajmc.ocr.font_utils.Font,
            'fallback': [ajmc.ocr.font_utils.Font]}`.
            Notice that ``'fallback'`` must be a list of ``Font``.
        fallback_fonts (list): A list of fallback fonts to use when a can't be printed with the main font.
        target_height (int): The desired height of the output image, in pixels.
        font_variants (list): A list of size len(text) containing the font variant to use for each character,
            eg. ``['Regular', 'Bold', 'Italic', 'Regular']``.
        kerning (int): The kerning to apply to the text, in pixels.
        default_charset (str): The default font to use when a character doesn't belong to any charset.
        output_file (Path): The path to save the image to.
        show_image (bool): Whether to show the image.
    """

    # Set default values
    upscale_factor = 3
    kerning *= upscale_factor

    # Get the drawboard height
    drawboard_height: int = target_height * upscale_factor
    font_size = int(0.5 * drawboard_height)

    # Adapt the font size to the text
    for charset, font in fonts.items():
        font.set_size(font_size)

    for font in fallback_fonts:
        font.set_size(font_size)

    # Chunk the text
    chars = [(char, get_char_charset(char, fallback=default_charset), variant) for char, variant in zip(textline, font_variants)]

    # Get the text sizetypeface
    drawboard_width = int(sum([fonts[charset].pil_font.getlength(char) for char, charset, _ in chars]))
    drawboard_width += int(2 * drawboard_width + (kerning * len(chars)))  # We take some leeway (we crop afterwards)

    # Create a drawboard
    drawboard = Image.new('L', (drawboard_width, drawboard_height), color='black')
    draw = ImageDraw.Draw(drawboard)

    # Draw the text, chunk by chunk and charset by charset
    x = 0
    y = 3 / 4 * drawboard_height

    def draw_char(char, font: Font, x, y):
        draw.text(xy=(x, y), text=char, font=font.pil_font, fill='white', anchor='ls')
        x += font.pil_font.getlength(char) + kerning
        return x

    for i, (char, charset, variant) in enumerate(chars):

        font = fonts[charset].font_variants.get(variant, fonts[charset])

        if font.has_glyph(char):
            x = draw_char(char, font, x, y)

        else:
            for fallback_font in fallback_fonts:
                if fallback_font.has_glyph(char):
                    logger.debug(f'Char {char} could not be displayed by {fonts[charset].path.stem} font, using {fallback_font.name}')
                    x = draw_char(char, fallback_font, x, y)
                    break
            else:
                raise ValueError(f'Char {char} could be displayed by any font')

    # Crop the drawboard to the text, resize it to the desired height, and paste it on a new image, adding padding
    crop = drawboard.crop(drawboard.getbbox())
    resize_factor = target_height / crop.height
    target_width = int(resize_factor * crop.width)
    final_image = crop.resize((target_width, target_height))
    final_image = ImageOps.invert(final_image)

    if show_image:
        final_image.show()

    if output_file:
        final_image.save(output_file)

    # print(f'Drew text in {time.time() - start_time} seconds')
    return final_image


def pad_image(image: Image,
              padding: Tuple[int, int, int, int],
              target_height: Optional[int] = None,
              background_color: str = 'black') -> Image:
    """

    Args:
        image (PILImage): The image to pad.
        padding (tuple): The padding to apply, in the form (top, right, bottom, left).
        target_height (int): The desired height of the output image, in pixels.
        If specified, ``image`` will be resized so that its height is ``target_height`` after padding.
        background_color (str): The color of the background.

    Returns:
        PILImage: The padded image.
    """

    if target_height is None:
        padded_image = Image.new('L', (image.width + padding[1] + padding[3], image.height + padding[0] + padding[2]), color=background_color)
        padded_image.paste(image, box=(padding[3], padding[0]))

    else:
        inner_height = target_height - padding[0] - padding[2]
        resize_factor = inner_height / image.height
        inner_width = int(resize_factor * image.width)
        resized_image = image.resize((inner_width, inner_height))
        padded_image = pad_image(resized_image, padding, background_color=background_color)

    return padded_image


def get_textline_font_variants(textline: str,
                               default_variant: str = 'Regular') -> List[str]:
    """A very custom function to generate font variants for a textline.

    Args:
        textline (str): The textline to generate font variants for.
        default_variant (str, optional): The default font variant to use. Defaults to 'Regular'.

    Returns:
        List[str]: A list of font variants, one for each character in the textline.
    """

    def select_words(k, line_words, variants_list):
        selected_words_offsets = [(textline.index(w), textline.index(w) + len(w)) for w in random.sample(line_words, k)]

        for offsets in selected_words_offsets:
            word_variant = random.sample(['Bold', 'Italic'], 1)[0]
            for i in range(offsets[0], offsets[1]):
                variants_list[i] = word_variant

        return variants_list

    a = random.randint(0, 100)
    line_words = textline.split()
    variants_list = [default_variant] * len(textline)

    if a == 0:  # 1% chance of bolding the whole line
        return ['Bold'] * len(textline)

    elif a < 3:  # 2% chance of italicising the whole line
        return ['Italic'] * len(textline)

    elif a < 12:  # 9% chance of bolding or italicising a word
        return select_words(k=1, line_words=line_words, variants_list=variants_list)

    elif len(line_words) >= 6 and a <= 30:  # ~6% chance of bolding or italicising two or three words
        return select_words(k=random.randint(2, 3), line_words=line_words, variants_list=variants_list)

    return variants_list


def textline_generator(text: List[str],
                       offset: int,
                       line_length_distribution: Dict[int, float]):
    """Loops infinitely over a text to generate textlines from it.

    Args:
    text (List[str]): The text to generate textlines from, as a list of words
    """

    length = len(text)

    while True:
        line_length = random.choices(population=list(line_length_distribution.keys()),
                                     weights=list(line_length_distribution.values()))
        line_length = line_length[0]
        yield ' '.join(text[offset:offset + line_length])
        offset += line_length
        if offset >= length:
            offset = 0


def get_textline_generators(corpus_ids: List[str],
                            line_length_distribution: Dict[int, float]) -> Dict[str, Generator[str, None, None]]:
    textline_generators = {}
    for corpus_id in corpus_ids:
        text = Corpus.auto_init(corpus_id).get_plain_text().split(' ')
        textline_generators[corpus_id] = textline_generator(text, offset=random.randint(0, len(text)),
                                                            line_length_distribution=line_length_distribution)

    logger.debug('Got textline generators')

    return textline_generators


def inner_loop(line_count_per_font: int,
               fonts: Dict[str, Font],
               fallback_fonts: List[Font],
               textline_generators: Dict[str, Generator[str, None, None]],
               fonts_distribution: Dict[str, float],
               kerning_distribution: Dict[int, float],
               target_height: int,
               output_dir: Path,
               charset: Optional[str] = None,
               mixed_charsets: Optional[Dict[str, float]] = None,
               base_font_charset: str = 'latin',
               capitalize: bool = False,
               add_numbers: bool = False, ) -> None:
    """Inner loop of the script."""
    file_number = len(list(output_dir.glob('*.png')))
    lines_per_corpus = line_count_per_font * fonts_distribution[fonts[base_font_charset].path.name] / len(textline_generators)
    lines_per_corpus = int(lines_per_corpus)

    for corpus_id, textline_generator in textline_generators.items():
        corpus_lines = 0

        while corpus_lines < lines_per_corpus:

            textline = next(textline_generator)

            # Normalize the textline and remove unknown characters
            textline = unicode_utils.harmonise_unicode(textline)
            textline = unicodedata.normalize('NFD', textline)
            missing_chars = [char for char in textline if char not in ALL_AJMC_CHARS_NFD]

            if len(missing_chars) > min(0.1 * len(textline), 3):  # Remove lines with more than 3 missing characters
                logger.debug(f'Skipping line    {textline}')
                logger.debug(f'Because of chars {missing_chars}')
                continue

            textline = ''.join([c for c in textline if c not in missing_chars])
            textline = re.sub(r'\s+', ' ', textline)  # Remove multiple spaces

            if not any(c.isalnum() for c in textline):  # Skip lines with only spaces and punctuation
                # logger.debug(f'Skipping empty line')
                continue

            # Skip lines with too many characters outside the given charset
            if charset is not None:
                if not unicode_utils.is_charset_string_nfd(textline, charset, threshold=0.9, strict=False):
                    logger.debug(f'Too many non-{charset} chars in :  {textline}')
                    logger.debug(f'Chars: {set(char for char in textline if char not in unicode_utils.CHARSETS_CHARS_NFD[charset])}')
                    continue

            if mixed_charsets is not None:
                for charset_, threshold in mixed_charsets.items():
                    if not unicode_utils.is_charset_string_nfd(textline, charset_, threshold=threshold, strict=True):
                        logger.debug(f'Mixed charsets conditio not satisfied in :  {textline}')
                        continue

            if capitalize:
                textline = textline.upper()
                textline = textline.replace('͂', '')  # Remove the perispomeni accent which is buggy with caps

            if add_numbers:
                if random.randint(1, 100) < 5:  # 4% chance of adding a number
                    number = ''.join(random.choices(population=list('0123456789'), k=random.randint(1, 4)))
                    split = textline.split(' ')
                    split.insert(random.randint(0, len(split)), number)
                    textline = ' '.join(split)

            # Draw the line and save it
            textline = unicodedata.normalize('NFC', textline)
            kerning = random.choices(list(kerning_distribution.keys()), list(kerning_distribution.values()))[0]
            char_variants = get_textline_font_variants(textline)
            bold = char_variants.count('Bold')
            italic = char_variants.count('Italic')
            file_name = f'{fonts["latin"].path.stem.split("-")[0]}_{fonts["greek"].path.stem.split("-")[0]}_{corpus_id}_k{kerning}_bold{bold}_ital{italic}_{int_to_x_based_code(file_number, fixed_min_len=4)}.png'
            output_file = output_dir / file_name

            try:
                draw_textline(textline=textline,
                              fonts=fonts,
                              fallback_fonts=fallback_fonts,
                              target_height=target_height,
                              font_variants=char_variants,
                              kerning=kerning,
                              output_file=output_file,
                              show_image=False)


            except Exception as e:
                logger.debug(f'Error while drawing {textline}')
                logger.debug(e)
                continue

            # Save the text as well
            output_file.with_suffix('.txt').write_text(textline, encoding='utf-8')
            corpus_lines += 1
            file_number += 1


def do_all_purpose_fonts(line_count_per_font=200,
                         add_numbers=True):
    logger.info('Starting all-purpose fonts')
    output_dir = BASE_OUTPUT_DIR / 'all_purpose_fonts'
    output_dir.mkdir(exist_ok=True, parents=True)

    corpus_ids = [
        # 'First1KGreek',
        'corpus_thomisticum',
        'logeion_latin',
        'logeion_greek',
        # 'agoraclass',
        # 'corpus_scriptorum_latinorum',
        'forum_romanum',
        # 'mediterranee_antique',
        'remacle',
        # 'the_latin_library',
        'perseus_secondary',
        'canonical-latinLit',
        'canonical-greekLit'
    ]

    textline_generators = get_textline_generators(corpus_ids, line_length_distribution=LINE_LENGTH_DISTRIBUTION)

    # textline_generators = {'lorem_ipsum': textline_generator(LOREM_IPSUM, offset=random.randint(0, len(LOREM_IPSUM)),
    #                                                          line_length_distribution=LINE_LENGTH_DISTRIBUTION)}

    for font in tqdm(FONTS_GROUPS['all_purpose']):
        fonts = {charset: font for charset in ['latin', 'greek', 'numeral', 'punctuation']}

        inner_loop(line_count_per_font=line_count_per_font,
                   fonts=fonts,
                   fallback_fonts=FONTS_GROUPS['fallback'],
                   textline_generators=textline_generators,
                   fonts_distribution=FONTS_DISTRIBUTION,
                   kerning_distribution=KERNING_DISTRIBUTION,
                   target_height=TARGET_HEIGHT,
                   output_dir=output_dir,
                   add_numbers=add_numbers)


def do_latin_fonts(line_count_per_font=200,
                   add_numbers=True):
    logger.info('Starting latin fonts')
    output_dir = BASE_OUTPUT_DIR / 'latin_fonts'
    output_dir.mkdir(exist_ok=True, parents=True)

    corpus_ids = ['corpus_thomisticum',
                  'logeion_latin',
                  'agoraclass',
                  'corpus_scriptorum_latinorum',
                  'forum_romanum',
                  'remacle',
                  'the_latin_library',
                  'canonical-latinLit']

    textline_generators = get_textline_generators(corpus_ids, line_length_distribution=LINE_LENGTH_DISTRIBUTION)

    for font in tqdm(FONTS_GROUPS['latin']):
        fonts = {charset: font for charset in ['latin', 'greek', 'numeral', 'punctuation']}

        inner_loop(line_count_per_font=line_count_per_font,
                   fonts=fonts,
                   fallback_fonts=FONTS_GROUPS['fallback'],
                   textline_generators=textline_generators,
                   charset='latin',
                   output_dir=output_dir,
                   fonts_distribution=FONTS_DISTRIBUTION,
                   kerning_distribution=KERNING_DISTRIBUTION,
                   target_height=TARGET_HEIGHT,
                   add_numbers=add_numbers)


def do_greek_fonts(line_count_per_font=500):
    logger.info('Starting greek fonts')
    output_dir = BASE_OUTPUT_DIR / 'greek_fonts'
    output_dir.mkdir(exist_ok=True, parents=True)

    corpus_ids = [
        'canonical-greekLit']

    textline_generators = get_textline_generators(corpus_ids, line_length_distribution=LINE_LENGTH_DISTRIBUTION)

    for font in tqdm(FONTS_GROUPS['greek']):
        fonts = {charset: font for charset in ['latin', 'greek', 'numeral', 'punctuation']}

        inner_loop(line_count_per_font=line_count_per_font,
                   fonts=fonts,
                   fallback_fonts=FONTS_GROUPS['fallback'],
                   textline_generators=textline_generators,
                   charset='greek',
                   output_dir=output_dir,
                   base_font_charset='greek',
                   fonts_distribution=FONTS_DISTRIBUTION,
                   kerning_distribution=KERNING_DISTRIBUTION,
                   target_height=TARGET_HEIGHT)


def do_mixed_fonts(line_count_per_font=25, add_numbers=True):
    logger.info('Starting mixed fonts')
    output_dir = BASE_OUTPUT_DIR / 'mixed_fonts'
    output_dir.mkdir(exist_ok=True, parents=True)

    corpus_ids = ['logeion_greek',
                  'perseus_secondary']

    textline_generators = get_textline_generators(corpus_ids, line_length_distribution={7: 0.2, 8: 0.3, 9: 0.3, 10: 0.2})

    for greek_font in tqdm(FONTS_GROUPS['greek']):
        for latin_font in tqdm(FONTS_GROUPS['latin']):
            fonts = {'latin': latin_font, 'greek': greek_font, 'numeral': latin_font, 'punctuation': latin_font}

            inner_loop(line_count_per_font=line_count_per_font,
                       fonts=fonts,
                       fallback_fonts=FONTS_GROUPS['fallback'],
                       textline_generators=textline_generators,
                       output_dir=output_dir,
                       base_font_charset='greek',
                       fonts_distribution=FONTS_DISTRIBUTION,
                       kerning_distribution=KERNING_DISTRIBUTION,
                       target_height=TARGET_HEIGHT,
                       mixed_charsets={'latin': 0.20, 'greek': 0.15},
                       add_numbers=add_numbers)


def do_capitals(line_count_per_font=200):
    logger.info('Starting capital only fonts')
    output_dir = BASE_OUTPUT_DIR / 'capitals'
    output_dir.mkdir(exist_ok=True, parents=True)

    corpus_ids = ['canonical-greekLit']

    textline_generators = get_textline_generators(corpus_ids, line_length_distribution={1: 0.4, 2: 0.4, 3: 0.1, 4: 0.05, 5: 0.05})

    for font in tqdm(FONTS_GROUPS['greek_capitals']):
        fonts = {charset: font for charset in ['latin', 'greek', 'numeral', 'punctuation']}

        inner_loop(line_count_per_font=line_count_per_font,
                   fonts=fonts,
                   fallback_fonts=FONTS_GROUPS['fallback'],
                   textline_generators=textline_generators,
                   charset='greek',
                   output_dir=output_dir,
                   base_font_charset='greek',
                   capitalize=True,
                   fonts_distribution=FONTS_DISTRIBUTION,
                   kerning_distribution={0: 0.4, 1: 0.4, 2: 0.1, 3: 0.05, 4: 0.05},
                   target_height=TARGET_HEIGHT,
                   )

    corpus_ids = ['corpus_thomisticum']

    textline_generators = get_textline_generators(corpus_ids, line_length_distribution={1: 0.4, 2: 0.4, 3: 0.1, 4: 0.05, 5: 0.05})

    for font in tqdm(FONTS_GROUPS['latin_capitals']):
        fonts = {charset: font for charset in ['latin', 'greek', 'numeral', 'punctuation']}

        inner_loop(line_count_per_font=line_count_per_font,
                   fonts=fonts,
                   fallback_fonts=FONTS_GROUPS['fallback'],
                   textline_generators=textline_generators,
                   charset='latin',
                   output_dir=output_dir,
                   base_font_charset='latin',
                   capitalize=True,
                   fonts_distribution=FONTS_DISTRIBUTION,
                   kerning_distribution={0: 0.4, 1: 0.4, 2: 0.1, 3: 0.05, 4: 0.05},
                   target_height=TARGET_HEIGHT,
                   )


def do_gibberish(line_count_per_font=5):
    logger.info('Starting gibberish')
    output_dir = BASE_OUTPUT_DIR / 'gibberish'
    output_dir.mkdir(exist_ok=True, parents=True)

    source_chars = ''.join([char for charset, chars in unicode_utils.CHARSETS_CHARS_NFC.items() for char in chars])
    source_chars = ''.join([char for char in source_chars if all([c in ALL_AJMC_CHARS_NFD for c in unicodedata.normalize('NFD', char)])])

    # strip accents
    source_chars = ''.join([char for char in source_chars if not unicode_utils.get_char_unicode_name(char).startswith('COMBINING')])

    source_text = []
    for i in range(1000):
        k = random.randint(1, 7)
        source_text.append(''.join(random.sample(source_chars, k)))

    textline_generators = {'gibberish': textline_generator(text=source_text, offset=0, line_length_distribution=LINE_LENGTH_DISTRIBUTION)}

    for font in tqdm(FONTS):
        fonts = {charset: font for charset in ['latin', 'greek', 'numeral', 'punctuation']}

        inner_loop(line_count_per_font=line_count_per_font,
                   fonts=fonts,
                   fallback_fonts=FONTS_GROUPS['fallback'],
                   textline_generators=textline_generators,
                   output_dir=output_dir,
                   base_font_charset='greek',
                   capitalize=False,
                   fonts_distribution=FONTS_DISTRIBUTION,
                   kerning_distribution=KERNING_DISTRIBUTION,
                   target_height=TARGET_HEIGHT,
                   )


def create_sample_dir():
    main_dir = Path('/scratch/sven/ocr_exp/source_datasets/artificial_data/augmented')
    test_dir = Path('/scratch/sven/ocr_exp/source_datasets/artificial_data_test_augmented')

    shutil.rmtree(test_dir, ignore_errors=True)

    for dir_ in main_dir.iterdir():
        if dir_.is_dir() and dir_.name:
            files = list(dir_.glob('*.png'))
            files = random.sample(files, k=min(50, len(files)))
            test_subdir = test_dir / dir_.name
            test_subdir.mkdir(exist_ok=True, parents=True)
            for file in files:
                (test_subdir / file.name).write_bytes(file.read_bytes())
                # (test_subdir / file.with_suffix('.txt').name).write_text(file.with_suffix('.txt').read_text('utf-8'), encoding='utf-8')


def check_file_count():
    for dir_ in BASE_OUTPUT_DIR.iterdir():
        if dir_.is_dir():
            print(dir_.name, len(list(dir_.glob('*.png'))))



#%%

# shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
# BASE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
# random.seed(0)
# do_all_purpose_fonts(2, True)
# do_latin_fonts(3)
# do_greek_fonts(3)
# do_capitals(1)
# do_mixed_fonts(1)
# do_gibberish(1)

#%%
if __name__ == '__main__':
    # all_purpose(10): 3300 images --> 1000 = 300000
    # latin(10): 5400 images --> 550 = 300000
    # greek(10): 800 images --> 3750 = 300000
    # mixed(10): 9000 images --> 333 = 300000
    # capitals(10): 500 images --> 1000 = 50000
    # gibberish(1): 1200 images --> 30 = 35000
    # Get the distributions for line length, kerning and fonts
    LINE_LENGTH_DISTRIBUTION = json.loads(Path('./data/fonts/glyphs/data_generation_config/line_length_distribution.json').read_text())
    LINE_LENGTH_DISTRIBUTION = {int(k): v for k, v in LINE_LENGTH_DISTRIBUTION.items()}

    KERNING_DISTRIBUTION = json.loads(Path('./data/fonts/glyphs/data_generation_config/kerning_distribution.json').read_text())
    KERNING_DISTRIBUTION = {int(k): v for k, v in KERNING_DISTRIBUTION.items()}

    FONTS_DISTRIBUTION = json.loads(Path('./data/fonts/glyphs/data_generation_config/fonts_distribution.json').read_text())

    TARGET_HEIGHT = 80

    ALL_AJMC_CHARS_NFD = Path('./data/fonts/glyphs/ajmc_final_glyphs.json').read_text(encoding='utf-8')
    ALL_AJMC_CHARS_NFD = ''.join(json.loads(ALL_AJMC_CHARS_NFD).keys())  # This is NFD

    # BASE_OUTPUT_DIR = Path('/Users/sven/Desktop/coucou')
    BASE_OUTPUT_DIR = Path('/scratch/sven/ocr_exp/source_datasets/artificial_data')

    FONTS_DIR = Path('./data/fonts/fonts')
    FONTS = [Font(font_path, font_variant='Regular') for font_path in walk_through_font_dir(FONTS_DIR)]

    GREEK_CAPITAL_FONTS_NAMES = ["GFSIgnacio-Regular.otf",
                                 "GFSGaraldus-Regular.otf",
                                 "GFSAmbrosia-Regular.otf",
                                 "GFSEustace-Regular.otf",
                                 "GFSFleischman-Regular.otf",
                                 "GFSJackson-Regular.otf",
                                 "GFSNicefore-Regular.otf",
                                 "Porson-Regular.otf",
                                 ]

    LATIN_CAPITAL_FONTS_NAMES = ["EBGaramond-Regular.otf",
                                 "Didot-Regular.otf",
                                 "TimesNewRoman-Regular.ttf", ]

    FONTS_GROUPS = {'all_purpose': [font for font in FONTS if font.has_charset_glyphs('latin', 20) and font.has_charset_glyphs('greek', 20)],
                    'latin': [font for font in FONTS if font.has_charset_glyphs('latin', 20) and not font.has_charset_glyphs('greek', 20)],
                    'greek': [font for font in FONTS if font.has_charset_glyphs('greek', 30) and not font.has_charset_glyphs('latin', 20)],
                    'greek_capitals': [font for font in FONTS if font.path.name in GREEK_CAPITAL_FONTS_NAMES],
                    'latin_capitals': [font for font in FONTS if font.path.name in LATIN_CAPITAL_FONTS_NAMES],
                    'fallback': [Font(Path('./data/fonts/fonts/NotoSans-Regular.ttf'), font_variant='Regular'),
                                 Font(Path('./data/fonts/fonts/Cardo-Regular.ttf'), font_variant='Regular'), ]}

    parser = argparse.ArgumentParser()
    parser.add_argument('--all_purpose', action='store_true')
    parser.add_argument('--latin', action='store_true')  #
    parser.add_argument('--greek', action='store_true')
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--capitals', action='store_true')
    parser.add_argument('--gibberish', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    # BASE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    if args.all_purpose:
        do_all_purpose_fonts(1000)
    if args.latin:
        do_latin_fonts(550)
    if args.greek:
        do_greek_fonts(1250)
    if args.mixed:
        do_mixed_fonts(300)
    if args.capitals:
        do_capitals(1500)
    if args.gibberish:
        do_gibberish(60)

