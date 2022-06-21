"""This module contains sample objects which are sent to `sample_objects.json` and used as fixtures elsewhere."""

from ajmc.nlp.token_classification.evaluation import seqeval_evaluation
from ajmc.commons import variables
import os
from ajmc.text_importation.classes import Commentary

# Commentaries, OCR, path and via
sample_base_dir = variables.PATHS['base_dir']

sample_commentary_id = 'cu31924087948174'
sample_page_id = sample_commentary_id + '_0083'

sample_via_path = os.path.join(sample_base_dir, sample_commentary_id, variables.PATHS['via_path'])

sample_ocr_run = '2480ei_greek-english_porson_sophoclesplaysa05campgoog'

sample_ocr_dir = os.path.join(sample_base_dir, sample_commentary_id, variables.PATHS['ocr'], sample_ocr_run, 'outputs')
sample_ocr_page_path = os.path.join(sample_ocr_dir, sample_page_id + '.hocr')

sample_groundtruth_dir = os.path.join(sample_base_dir, sample_commentary_id, variables.PATHS['groundtruth'])
sample_groundtruth_page_path = os.path.join(sample_groundtruth_dir, sample_page_id + '.hmtl')

sample_image_dir = os.path.join(sample_base_dir, sample_commentary_id, variables.PATHS['png'])
sample_image_path = os.path.join(sample_image_dir, sample_page_id + '.png')

sample_commentary = Commentary.from_folder_structure(sample_ocr_dir)
sample_page = [p for p in sample_commentary.pages if p.id == sample_page_id][0]



# NLP, NER...
from transformers import DistilBertTokenizerFast

sample_ner_labels_pred = ['O', 'B-PERS', 'I-PERS', 'B-LOC', 'O']
sample_ner_labels_gt = ['O', 'B-PERS', 'I-PERS', 'B-LOC', 'I-LOC']

sample_seqeval_output = seqeval_evaluation([sample_ner_labels_pred],
                                           [sample_ner_labels_gt])

sample_model_name_or_path = 'distilbert-base-uncased'
sample_tokenizer = DistilBertTokenizerFast.from_pretrained(sample_model_name_or_path)

sample_encodings = sample_tokenizer(text=[w.text for w in sample_page.words],
                                    truncation=True,
                                    padding=True,
                                    return_overflowing_tokens=True,
                                    is_split_into_words=True)


