import os
from ajmc.commons.miscellaneous import docstring_formatter
from ajmc.commons.docstrings import docstrings
import numpy as np
import torch
from torch.utils.data import RandomSampler
from transformers import LayoutLMv2Tokenizer, LayoutLMv2ForTokenClassification, LayoutLMv2FeatureExtractor
from typing import List, Optional, Dict, Any
from ajmc.nlp.token_classification.pipeline import create_dirs, train
from ajmc.text_importation.classes import Commentary
from PIL import Image
from ajmc.commons.miscellaneous import read_google_sheet
from ajmc.olr.layout_lm.config import create_olr_config


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
    for x in range(0, len(list_), n):
        chunk = list_[x: n + x]

        if len(chunk) < n:
            chunk += [pad for _ in range(n - len(chunk))]

        chunks.append(chunk)

    return chunks


class LayoutLMDataset(torch.utils.data.Dataset):

    def __init__(self,
                 input_ids: List[List[int]],
                 bbox: List[List[List[int]]],
                 token_type_ids: List[List[int]],
                 attention_mask: List[List[int]],
                 image: List[np.ndarray],
                 labels: Optional[List[List[int]]] = None):
        self.input_ids = input_ids
        self.bbox = bbox
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.image = image
        self.labels = labels

    def __getitem__(self, idx):
        item = {'input_ids': torch.tensor(self.input_ids[idx]),
                'bbox': torch.tensor(self.bbox[idx]),
                'token_type_ids': torch.tensor(self.token_type_ids[idx]),
                'attention_mask': torch.tensor(self.attention_mask[idx]),
                'image': torch.tensor(self.image[idx])}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.input_ids)


# Get pages
def get_pages(ocr_output_dirs: List[str], splits: List[str]) -> Dict[str, List['Page']]:
    split_pages = {s: [] for s in splits}

    for ocr_output_dir in ocr_output_dirs:
        commentary = Commentary.from_folder_structure(ocr_dir=ocr_output_dir)

        # Read sheet  # Todo : create a variable for this
        sheet_id = '1_hDP_bGDNuqTPreinGS9-ShnXuXCjDaEbz-qEMUSito'
        olr_gt = read_google_sheet(sheet_id, 'olr_gt')

        split_ids = {s: list(olr_gt.loc[(olr_gt['split'] == s) & (olr_gt['id'] == commentary.id)]['page_id']) for s in
                     splits}
        for split in splits:
            split_pages[split] += [p for p in commentary.pages if p.id in split_ids[split]]

    return split_pages


@docstring_formatter(**docstrings)
def prepare_data(split_pages: Dict[str, List['Page']],
                 model_inputs: List[str],
                 labels_to_ids: Dict[str, int],
                 regions_to_coarse_labels: Dict[str, str],
                 rois: List[str],
                 special_tokens: Dict[str, Dict[str, Any]],
                 tokenizer,
                 feature_extractor,
                 max_length: int = 512,
                 do_debug: bool = False
                 ) -> Dict[str, LayoutLMDataset]:
    """Prepares data for LayoutLMV2.

    Args:
        split_pages: A dict containing a list of `Page`s per split.
        model_inputs: List of inputs the model wants (inputs_ids, attention_mask,...)
        labels_to_ids: {labels_to_ids}
        regions_to_coarse_labels:
        rois: The regions to focus on
        special_tokens: e.g. `{{'start': {{'input_ids':100, ...}}, ...}}`
        tokenizer:
        feature_extractor:
        max_length: {max_length}
        do_debug:

    """

    encodings = {s: {k: [] for k in model_inputs} for s in split_pages.keys()}

    for s in split_pages.keys():
        for i, page in enumerate(split_pages[s]):

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
                        word_labels.append(labels_to_ids[regions_to_coarse_labels[r.region_type]])

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
                tokens[k] = split_list(tokens[k], max_length - 2,
                                       special_tokens['pad'][k])  # We divide our list in lists of len 510
                tokens[k] = [[special_tokens['start'][k]] + ex + [special_tokens['end'][k]] for ex in tokens[k]]

            # We create a list of input dicts
            image = Image.open(page.image.path).convert('RGB')
            image = feature_extractor(image)['pixel_values']

            # Appending to encodings
            for i in range(len(tokens['input_ids'])):
                for k in tokens.keys():
                    encodings[s][k].append(tokens[k][i])
                encodings[s]['image'].append(image[0].copy())

            if i == 2 and do_debug:
                break

    return {s: LayoutLMDataset(**encodings[s]) for s in split_pages.keys()}


def main(config):
    pass


config = create_olr_config('/Users/sven/packages/ajmc/data/configs/simple_config_local.json')
create_dirs(config)

tokenizer = LayoutLMv2Tokenizer.from_pretrained(config.model_name_or_path)
feature_extractor = LayoutLMv2FeatureExtractor.from_pretrained(config.model_name_or_path, apply_ocr=False)
model = LayoutLMv2ForTokenClassification.from_pretrained(config.model_name_or_path, num_labels=config.num_labels)

pages = get_pages(ocr_output_dirs=config.ocr_dirs, splits=config.splits)
datasets = prepare_data(split_pages=pages,
                        model_inputs=config.model_inputs,
                        labels_to_ids=config.labels_to_ids,
                        regions_to_coarse_labels=config.regions_to_coarse_labels,
                        rois=config.rois,
                        special_tokens=config.special_tokens,
                        tokenizer=tokenizer,
                        feature_extractor=feature_extractor,
                        do_debug=config.do_debug)

train(config=config, model=model, train_dataset=datasets['train'], eval_dataset=datasets['dev'],
      tokenizer=tokenizer)
