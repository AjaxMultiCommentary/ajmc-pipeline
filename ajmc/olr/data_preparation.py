import os
import torch
from transformers import LayoutLMv2Processor, LayoutLMv2Tokenizer, LayoutLMv2ForTokenClassification, \
    LayoutLMv2FeatureExtractor
from transformers import LayoutXLMProcessor, LayoutXLMTokenizer, \
    LayoutLMv2FeatureExtractor  # LayoutCLM is NOT implemented for token classification 😩
from typing import List, Any
from ajmc.text_importation.classes import Commentary
from PIL import Image
from ajmc.commons import variables

# Variables

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

# Special tokens values
special_tokens = {
    'start': {'input_ids': 101, 'bbox': [0, 0, 0, 0], 'token_type_ids': 0, 'labels': -100, 'attention_mask': 1},
    'end': {'input_ids': 102, 'bbox': [1000, 1000, 1000, 1000], 'token_type_ids': 0, 'labels': -100,
            'attention_mask': 1},
    'pad': {'input_ids': 0, 'bbox': [0, 0, 0, 0], 'token_type_ids': 0, 'labels': -100, 'attention_mask': 0},
}

params = {'splits': ['train','dev'],
          'batch_size': 1,
          'model_inputs': ['input_ids', 'bbox', 'token_type_ids', 'labels', 'attention_mask', 'image'],
          'max_length': 512}

# Functions
def normalize_bounding_rectangles(rectangle: List[List[int]], img_width: int, img_height: int, ):
    return [
        int(1000 * (rectangle[0][0] / img_width)),
        int(1000 * (rectangle[0][1] / img_height)),
        int(1000 * (rectangle[2][0] / img_width)),
        int(1000 * (rectangle[2][1] / img_height))
    ]


def split_list(list_: list, n: int, pad: object) -> List[List[object]]:
    """Divides a list into a list of lists with n elements, padding the last chunk with `pad`."""
    chunks = []
    for x in range(0, len(tokens[k]), n):
        chunk = list_[x: n + x]

        if len(chunk) < n:
            chunk += [pad for _ in range(n - len(chunk))]

        chunks.append(chunk)

    return chunks



# TODO : restart HERE============# TODO : restart HERE============# TODO : restart HERE============# TODO : restart HERE============# TODO : restart HERE============
class LayoutLMDataset(torch.utils.data.Dataset):

    def __init__(self,
                 batch_encoding: "BatchEncoding",
                 labels: List[List[int]],
                 tsv_line_numbers: List[List[int]],
                 words: List[List[str]]):
        self.batch_encoding = batch_encoding
        self.labels = labels
        self.tsv_line_numbers = tsv_line_numbers
        self.words = words
        self.token_offsets = [e.word_ids for e in batch_encoding.encodings]

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.batch_encoding.items() if k!='overflow_to_sample_mapping'}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# TODO : restart HERE============# TODO : restart HERE============# TODO : restart HERE============# TODO : restart HERE============# TODO : restart HERE============
# %% Script
tokenizer = LayoutLMv2Tokenizer.from_pretrained('microsoft/layoutlmv2-base-uncased')
processor = LayoutLMv2Processor.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                tokenizer=tokenizer,
                                                revision='no_ocr')
feature_extractor = LayoutLMv2FeatureExtractor.from_pretrained('microsoft/layoutlmv2-base-uncased', apply_ocr=False)
model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased', num_labels=6)

# %% Get pages
commentary = Commentary.from_folder_structure(
    ocr_dir=os.path.join(variables.PATHS['base_dir'], 'Wecklein1894/ocr/runs/13p0am_lace_base/outputs'))

from ajmc.commons.miscellaneous import read_google_sheet

sheet_id = '1_hDP_bGDNuqTPreinGS9-ShnXuXCjDaEbz-qEMUSito'
olr_gt = read_google_sheet(sheet_id, 'olr_gt')
#%%
split_ids = {s: list(olr_gt.loc[(olr_gt['split'] == s) & (olr_gt['id'] == commentary.id)]['page_id']) for s in params['splits']}
split_pages = {s: [p for p in commentary.pages if p.id in split_ids[s]] for s in params['splits']}


#%%
encodings = {s:[] for s in params['splits']}

for s in params['splits']:
    for page in split_pages[s]:

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
        # We tokenize without truncation, as LayoutLMV2Tokenizer does not handle overflowing tokens properly.
        tokens = tokenizer(text=words,
                           boxes=word_boxes,
                           word_labels=word_labels,
                           is_split_into_words=True)

        # We now do a manual truncation
        encoding = {}
        for k in tokens.keys():
            tokens[k] = tokens[k][1:-1]  # We start by triming the first and last tokens
            tokens[k] = split_list(tokens[k], params['max_length'] - 2, special_tokens['pad'][k]) # We divide our list in lists of len 510
            tokens[k] = [[special_tokens['start'][k]] + ex + [special_tokens['end'][k]] for ex in tokens[k]]

        # We create a list of input dicts
        image = Image.open(page.image.path).convert('RGB')
        image = feature_extractor(image, return_tensors='pt')['pixel_values']

        for i in range(len(tokens['input_ids'])):
            inputs = {k: torch.tensor([tokens[k][i]]) for k in
                      special_tokens['start'].keys()}  # ⚠️ adding a list in `[tokens[k][i]]` to create a simili batch
            inputs['image'] = image['pixel']

            encodings[s].append(inputs)



# %%
outputs = model(image=image['pixel_values'], **inputs_list[0])
# WORKS
