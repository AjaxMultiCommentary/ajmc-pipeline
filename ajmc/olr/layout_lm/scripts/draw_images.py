import os
import pandas as pd
import json
from ajmc.olr.utils import get_olr_split_page_ids
from ajmc.text_processing. canonical_classes import CanonicalCommentary
from transformers import LayoutLMv2TokenizerFast, LayoutLMv2ForTokenClassification
from ajmc.olr.layout_lm.config import rois, regions_to_coarse_labels, labels_to_ids, ids_to_labels
from ajmc.olr.layout_lm.layoutlm import draw_pages
from ajmc.commons.variables import COLORS



base_path = '/Users/sven/drive/layout_lm_tests'

results = pd.DataFrame()

for fname in next(os.walk(base_path))[1]:  # Walk in dirs only
    if not fname.startswith('z') and not fname=='all_tokens_b_ents':

        with open(os.path.join(base_path, fname, 'config.json'), "r") as file:
            config = json.loads(file.read())

        model_name_or_path = os.path.join(base_path, fname, 'model')

        pages = []
        old_prefix = '/content/drive/MyDrive/'
        new_prefix = '/Users/sven/drive/'
        for ocr_dir, splits in config['data_dirs_and_sets']['eval'].items():

            ocr_dir = os.path.join(new_prefix, ocr_dir[len(old_prefix):])
            output_dir = os.path.join(new_prefix, config['predictions_dir'][len(old_prefix):])
            commentary = CanonicalCommentary.from_json(ocr_dir)
            pages+= [p for p in commentary.children['page']
                     if p.id in get_olr_split_page_ids(commentary.id, splits)]

        tokenizer = LayoutLMv2TokenizerFast.from_pretrained(model_name_or_path)
        model = LayoutLMv2ForTokenClassification.from_pretrained(model_name_or_path)

        draw_pages(pages,rois=rois, labels_to_ids=labels_to_ids, ids_to_labels=ids_to_labels,
                   regions_to_coarse_labels=regions_to_coarse_labels, tokenizer=tokenizer,
                   model=model, output_dir=output_dir)

