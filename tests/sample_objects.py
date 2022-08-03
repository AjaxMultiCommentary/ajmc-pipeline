"""This module contains sample objects which are sent to `sample_objects.json` and used as fixtures elsewhere."""

from ajmc.nlp.token_classification.evaluation import seqeval_evaluation
from ajmc.commons import variables, geometry, image
import os
from ajmc.text_processing.ocr_classes import OcrCommentary
from ajmc.text_processing.canonical_classes import CanonicalCommentary

# Arithmetic
sample_intervals = {'base': (1, 10),
                    'included': (2, 8),
                    'overlapping': (8, 15),
                    'non_overlapping': (11, 20)}

sample_interval_lists = {'base': [(1, 10), (15, 15), (16, 20)],
                         'included': [(1, 10), (19, 20)],
                         'overlapping': [(2, 12)],
                         'non_overlapping': [(0, 0), (30, 40)]}

# Geometry
sample_points = {'base': [(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)],
                 'included': [(0, 0), (1, 0), (1, 1), (0, 1)],
                 'overlapping': [[1, 1], [3, 1], [2, 2], [3, 3], [1, 3]],
                 'non_overlapping': [(5, 5), (7, 5), (6, 6), (7, 7), (5, 7)],
                 'line': [(0, 0), (1, 1), (2, 2)],
                 'horizontally_overlapping': [(1, 0), (3, 0), (3, 2), (1, 2)],
                 }

sample_rectangles = {k: geometry.get_bbox_from_points(v) for k, v in sample_points.items()}

# Commentaries, OCR, path and via
sample_base_dir = "/Users/sven/packages/ajmc/data/sample_commentaries"

sample_commentary_id = 'cu31924087948174'
sample_page_id = sample_commentary_id + '_0083'

sample_via_path = os.path.join(sample_base_dir, sample_commentary_id, variables.PATHS['via_path'])

sample_ocr_run = 'tess_eng_grc'
sample_ocr_dir = os.path.join(sample_base_dir, sample_commentary_id, variables.PATHS['ocr'], sample_ocr_run, 'outputs')
sample_ocr_page_path = os.path.join(sample_ocr_dir, sample_page_id + '.hocr')

sample_groundtruth_dir = os.path.join(sample_base_dir, sample_commentary_id, variables.PATHS['groundtruth'])
sample_groundtruth_page_path = os.path.join(sample_groundtruth_dir, sample_page_id + '.hmtl')

sample_ocrcommentary = OcrCommentary.from_ajmc_structure(sample_ocr_dir)
sample_cancommentary = sample_ocrcommentary.to_canonical()

sample_cancommentary.to_json()
sample_canonical_path = os.path.join(sample_ocrcommentary.base_dir, 'canonical/v2', sample_ocrcommentary.ocr_run+'.json')

sample_cancommentary_from_json = CanonicalCommentary.from_json(sample_canonical_path)

sample_page = [p for p in sample_ocrcommentary.pages if p.id == sample_page_id][0]

# Image
sample_image_dir = os.path.join(sample_base_dir, sample_commentary_id, variables.PATHS['png'])
sample_image_path = os.path.join(sample_image_dir, sample_page_id + '.png')
sample_image = image.Image(id=sample_page_id, path=sample_image_path)

# NLP, NER...
from transformers import DistilBertTokenizerFast

sample_ner_labels_pred = ['O', 'B-PERS', 'I-PERS', 'B-LOC', 'O']
sample_ner_labels_gt = ['O', 'B-PERS', 'I-PERS', 'B-LOC', 'I-LOC']

sample_seqeval_output = seqeval_evaluation([sample_ner_labels_pred],
                                           [sample_ner_labels_gt])

sample_model_name_or_path = 'distilbert-base-uncased'
sample_tokenizer = DistilBertTokenizerFast.from_pretrained(sample_model_name_or_path)

sample_encodings = sample_tokenizer(text=[w.text for w in sample_page.children['word']],
                                    truncation=True,
                                    padding=True,
                                    return_overflowing_tokens=True,
                                    is_split_into_words=True)






