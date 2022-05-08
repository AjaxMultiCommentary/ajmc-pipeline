import os

from transformers import LayoutLMv2Processor, LayoutLMv2Tokenizer, LayoutLMv2ForTokenClassification, LayoutLMv2FeatureExtractor
from typing import List
from ajmc.text_importation.classes import Commentary
from PIL import Image
from ajmc.commons import variables

# %% Variables

# Select only the regions of interest we want
rois = ['app_crit',
        'appendix',
        'bibliography',
        'commentary',
        'footnote',
        'index_siglorum',
        'introduction',
        'line_number_text',
        # 'line_number_commentary',
        'printed_marginalia',
        # 'handwritten_marginalia',
        'page_number',
        'preface',
        'primary_text',
        'running_header',
        'table_of_contents',
        'title',
        'translation',
        'other',
        # 'undefined'
        ]

# Define our labels to ids
labels_to_ids = {
    # Commentary
    'commentary': 1,
    # Primary text
    'primary_text': 2,
    # Paratext
    'preface': 3,
    'introduction': 3,
    'footnote': 3,
    'appendix': 3,
    # Numbers
    'line_number_text': 4,
    'line_number_commentary': 4,
    'page_number': 4,
    # App Crit
    'app_crit': 5,
    # Others
    'translation': 0,
    'bibliography': 0,
    'index_siglorum': 0,
    'running_header': 0,
    'table_of_contents': 0,
    'title': 0,
    'printed_marginalia': 0,
    'handwritten_marginalia': 0,
    'other': 0,
    'undefined': 0
}


# %% Functions
def normalize_bounding_rectangles(rectangle: List[List[int]], img_width: int, img_height: int, ):
    return [
        int(1000 * (rectangle[0][0] / img_width)),
        int(1000 * (rectangle[0][1] / img_height)),
        int(1000 * (rectangle[2][0] / img_width)),
        int(1000 * (rectangle[2][1] / img_height))
    ]


# %% Script
commentary = Commentary.from_structure(ocr_dir=os.path.join(variables.PATHS['base_dir'], 'Wecklein1894/ocr/runs/15i0jT_ocrd_vanilla/outputs'))
tokenizer = LayoutLMv2Tokenizer.from_pretrained('microsoft/layoutlmv2-base-uncased')
processor = LayoutLMv2Processor.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                tokenizer=tokenizer,
                                                revision='no_ocr')
feature_extractor = LayoutLMv2FeatureExtractor.from_pretrained('microsoft/layoutlmv2-base-uncased')
model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased', num_labels=6)

# %%


encodings = []

# for page in commentary.olr_groundtruth_pages[0:33]:
page = commentary.olr_groundtruth_pages[6]

# Get the lists of words, boxes and labels for a single page
words = []
word_boxes = []
word_labels = []
for r in page.regions:
    if r.region_type in rois:
        for w in r.words:
            words.append(w.text)
            word_boxes.append(normalize_bounding_rectangles(w.coords.bounding_rectangle,
                                                            page.image.width,
                                                            page.image.height))
            word_labels.append(labels_to_ids[r.region_type])

# Tokenize, truncate and pad
tokens = tokenizer(words,
                   padding=True,
                   truncation=True,
                   is_split_into_words=True,
                   return_overflowing_tokens=True)

#%% Align labels and boxes
from ajmc.nlp.data_preparation.utils import align_elements

word_boxes = [align_elements(e.word_ids, word_boxes) for e in tokens.encodings]
word_labels = [align_elements(e.word_ids, word_labels) for e in tokens.encodings]

image = Image.open(page.image.path).convert('RGB')
image = feature_extractor(image)
images = []

encoding = processor(image, words, boxes=word_boxes, word_labels=word_labels, return_tensors="pt",
                     truncation=True, padding=True, return_overflowing_tokens=True)
encodings.append(encoding)

outputs = model(**{k: encoding[k] for k in encoding.keys() if k not in ['overflow_to_sample_mapping']})

# data[split]['batchencoding'] = tokenizer(data[split]['TOKEN'],
#                                          padding=True,
#                                          truncation=True,
#                                          is_split_into_words=True,
#                                          return_overflowing_tokens=True)
#
# data[split]['labels'] = [align_labels(e.word_ids, data[split][config.labels_column], config.labels_to_ids)
#                          for e in data[split]['batchencoding'].encodings]
#
# data[split]['words'] = [align_elements(e.word_ids, data[split]['TOKEN']) for e in
#                         data[split]['batchencoding'].encodings]
#
# data[split]['tsv_line_numbers'] = [align_elements(e.word_ids, data[split]['n']) for e in
#                                    data[split]['batchencoding'].encodings]
