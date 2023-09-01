from ajmc.ocr.pytorch.data_generation import draw_textline

#%% ############## CREATE THE DATASET ##############
# We start now to create the lines with a simple dataset
# 1. Creates the words
chars = '1aα'
words = {0: list(chars)}

for i in range(1, len(chars)):
    words[i] = []
    for j in words[i - 1]:
        for k in words[0]:
            words[i].append(j + k)

words = words[len(chars) - 1]

# 2. Creates the lines images


from pathlib import Path
import random

greek_fonts_dir = Path('/Users/sven/packages/ajmc/data/fonts/greek_fonts/gfs')
latin_fonts_dir = Path('/Users/sven/packages/ajmc/data/fonts/latin_fonts/most_common')

count_typo_combination = 0
i = 0
for greek_font_path in greek_fonts_dir.glob('*.*t*'):
    for latin_font_path in latin_fonts_dir.glob('*.*t*'):

        for word in words[i:i + 3]:
            split = 'test' if random.randint(0, 7) == 5 else 'train'
            draw_textline(textline=word, fonts={'latin': str(latin_font_path),
                                                'greek': str(greek_font_path),
                                                'numeral': str(latin_font_path),
                                                'punctuation': str(latin_font_path),
                                                'fallback': str(latin_font_path)}, fallback_fonts=, target_height=, output_file=Path(
                    f'/Users/sven/Desktop/pre_training_images_test/{split}/{greek_font_path.stem}_{latin_font_path.stem}_{word.replace("α", "A")}.png'))
        i += 4
        if i >= len(words):
            i -= len(words)

Path('/Users/sven/Desktop/pre_training_images_test/labels.txt').write_text('\n'.join(words).replace('α', 'A'))
