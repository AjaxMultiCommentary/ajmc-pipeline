from pathlib import Path
from typing import Dict, Iterable

from fontTools.ttLib import TTFont, TTCollection
from lazy_objects.lazy_objects import lazy_property
from PIL import ImageFont

from ajmc.commons import unicode_utils


def walk_through_font_dir(font_dir: Path):
    for font_path in font_dir.rglob('*.[ot]tf'):
        if font_path.stem.endswith('-Regular'):
            yield font_path


class Font:

    def __init__(self,
                 path: Path,
                 size: int = 12,
                 font_variant: str = 'Regular',
                 pil_encoding: str = 'unic',
                 pil_layout_engine: int = ImageFont.Layout.RAQM):

        """A wrapper for a font file.

        Args:
            path (Path): The path to the font file.
            size (int, optional): The size of the font, as used by ``PIL.ImageFont.truetype()``.
            font_variant (str, optional): For a``.ttc`` collection, the font variant to use. One of 'Bold', 'Italic', 'Regular'.
        """

        self.path = path
        self.size = size
        self.font_variant = font_variant.capitalize()
        self.pil_encoding = pil_encoding
        self.pil_layout_engine = pil_layout_engine


    @lazy_property
    def ft_collection(self):
        if self.path.suffix == '.ttc':
            return TTCollection(str(self.path))


    @lazy_property
    def ft_font(self):

        if self.ft_collection is not None:  # TTC file
            font_variant = None
            for font in self.ft_collection.fonts:
                try:
                    variant_name = font['name'].getName(2, 3, 1, 1033).toUnicode()
                except:
                    continue
                if variant_name == self.font_variant:
                    font_variant = font
                    break

            if font_variant is None:
                print(f'Font font_variant {self.font_variant} not found in TTC file {self.path}, using font_variant at index 0.')
                font_variant = self.ft_collection.fonts[0]

            return font_variant

        else:  # OTF or TTF file
            if self.font_variant in self.font_variants.keys():
                return TTFont(str(self.get_font_variant_path(self.font_variant)))
            else:
                print(f'Font font_variant {self.font_variant} not found in font file {self.path}, using Regular istead')
                return TTFont(str(self.get_font_variant_path('Regular')))

    @lazy_property
    def pil_font(self):
        return ImageFont.truetype(font=str(self.path),
                                  size=self.size,
                                  index=self.get_font_variant_index(),
                                  encoding=self.pil_encoding,
                                  layout_engine=self.pil_layout_engine)


    def has_glyph(self, glyph: str) -> bool:
        return any([ord(glyph) in table.cmap for table in self.ft_font['cmap'].tables])

    def has_glyphs(self, string: str, threshold: float = 1) -> bool:
        return len([glyph for glyph in string if self.has_glyph(glyph)]) / len(string) >= threshold

    def has_charset_glyphs(self, charset: str, missing_threshold: int = 0) -> bool:
        """Check if the font has glyphs for a given charset.

        Args:
            charset (str): The charset to check for. One of 'latin', 'greek', 'numeral' and 'punctuation'.
            missing_threshold (int, optional): The maximum number of missing glyphs allowed.
        """
        return len(self.get_missing_glyphs(unicode_utils.CHARSETS_CHARS_NFC[charset])) <= missing_threshold

    def get_missing_glyphs(self, string: Iterable) -> set:
        return set([glyph for glyph in string if not self.has_glyph(glyph)])

    def set_size(self, size: int):
        """Change the font size, resetting the PIL font."""
        for font in self.font_variants.values():
            font.size = size
            delattr(font, 'pil_font')


    @lazy_property
    def font_variants(self) -> Dict[str, 'Font']:

        font_variants = {}

        if self.ft_collection is not None:
            for font in self.ft_collection.fonts:
                try:
                    variant_name = font['name'].getName(2, 3, 1, 1033).toUnicode()
                    font_variants[variant_name] = self.__class__(self.path, self.size, variant_name, self.pil_encoding, self.pil_layout_engine)
                except:
                    continue
        else:
            for font_variant in ['Regular', 'Italic', 'Bold']:
                if self.get_font_variant_path(font_variant).exists():
                    if font_variant == self.font_variant:
                        font_variants[font_variant] = self
                    else:
                        font_variants[font_variant] = self.__class__(self.get_font_variant_path(font_variant),
                                                                     self.size, font_variant, self.pil_encoding, self.pil_layout_engine)

        return font_variants


    def get_font_variant_index(self):
        if self.ft_collection is not None:
            for i, font in enumerate(self.ft_collection.fonts):
                try:
                    if font['name'].getName(2, 3, 1, 1033).toUnicode() == self.font_variant:
                        return i
                except:
                    continue
            raise ValueError(f'Font font_variant {self.font_variant} not found in TTC file {self.path}.')

        else:
            return 0


    def get_font_variant_path(self, font_variant: str) -> Path:
        """Gets a font font_variant from a ttf or otf file."""
        if self.ft_collection is not None:
            return self.path
        else:
            font_variant = font_variant.capitalize()
            return self.path.with_name(self.path.stem.replace(f'-{self.font_variant}', f'-{font_variant}{self.path.suffix}'))

    @property
    def name(self):
        if self.ft_collection is not None:
            return self.path.stem
        else:
            return self.path.stem.replace(f'-{self.font_variant}', '')
