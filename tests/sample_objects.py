"""This module contains sample objects which are sent to ``sample_objects.json`` and used as fixtures elsewhere."""

from ajmc.commons import geometry, image, variables as vs
from ajmc.commons.miscellaneous import get_ajmc_logger, ROOT_LOGGER
from ajmc.text_processing.canonical_classes import CanonicalCommentary
from ajmc.text_processing.raw_classes import RawCommentary


ROOT_LOGGER.setLevel('DEBUG')
logger = get_ajmc_logger(__name__)


# Arithmetic
sample_intervals = {'base': (1, 10),
                    'included': (2, 8),
                    'overlapping': (8, 15),
                    'non_overlapping': (11, 20)}

sample_interval_lists = {'base': [(1, 10), (15, 15), (16, 20)],
                         'included': [(1, 10), (19, 20)],
                         'overlapping': [(2, 12)],
                         'non_overlapping': [(0, 0), (30, 40)]}

# Geometry tests
sample_points = {'base': [(0, 0), (2, 0), (1, 1), (2, 2), (0, 2)],
                 'included': [(0, 0), (1, 0), (1, 1), (0, 1)],
                 'overlapping': [[1, 1], [3, 1], [2, 2], [3, 3], [1, 3]],
                 'non_overlapping': [(5, 5), (7, 5), (6, 6), (7, 7), (5, 7)],
                 'line': [(0, 0), (1, 1), (2, 2)],
                 'horizontally_overlapping': [(1, 0), (3, 0), (3, 2), (1, 2)],
                 }

sample_bboxes = {k: geometry.get_bbox_from_points(v) for k, v in sample_points.items()}

# Commentaries, OCR, path and via


sample_comm_id = 'sample_test_cu31924087948174'

sample_comm_root_dir = vs.get_comm_root_dir(sample_comm_id)

sample_page_id = sample_comm_id + '_0083'

sample_via_path = vs.get_comm_via_path(sample_comm_id)

sample_ocr_run_id = '3464N4_tess_retrained'
sample_ocr_run_outputs_dir = vs.get_comm_ocr_outputs_dir(sample_comm_id, sample_ocr_run_id)
sample_ocr_page_path = sample_ocr_run_outputs_dir / (sample_page_id + '.hocr')

sample_img_dir = sample_comm_root_dir / vs.COMM_IMG_REL_DIR
sample_sections_path = sample_comm_root_dir / vs.COMM_SECTIONS_REL_PATH

sample_raw_commentary = RawCommentary(id=sample_comm_id,
                                      ocr_dir=sample_ocr_run_outputs_dir,
                                      base_dir=sample_comm_root_dir,
                                      via_path=sample_via_path,
                                      img_dir=sample_img_dir,
                                      ocr_run_id=sample_ocr_run_id,
                                      sections_path=sample_sections_path)

sample_ocr_page = sample_raw_commentary.get_page(sample_page_id)
sample_raw_entities = sample_ocr_page.children.entities

sample_can_commentary = sample_raw_commentary.to_canonical()

sample_canonical_path = vs.get_comm_canonical_path_from_ocr_run_pattern(sample_comm_id, sample_ocr_run_id)
sample_can_commentary.to_json(sample_canonical_path)

sample_cancommentary_from_json = CanonicalCommentary.from_json(sample_canonical_path)

sample_page = sample_can_commentary.get_page(sample_page_id)

# Image
sample_img_path = sample_img_dir / (sample_page_id + vs.DEFAULT_IMG_EXTENSION)
sample_img = image.AjmcImage(id=sample_page_id, path=sample_img_path)

# NLP, NER...
sample_ner_labels_pred = ['O', 'B-PERS', 'I-PERS', 'B-LOC', 'O']
sample_ner_labels_gt = ['O', 'B-PERS', 'I-PERS', 'B-LOC', 'I-LOC']
sample_labels_to_ids = {'O': 0, 'B-PERS': 1, 'I-PERS': 2, 'B-LOC': 3, 'I-LOC': 4}

# from transformers import DistilBertTokenizerFast
# from ajmc.nlp.token_classification.evaluation import seqeval_evaluation
# Uncomment this to work with transformers
# sample_seqeval_output = seqeval_evaluation([sample_ner_labels_pred],
#                                            [sample_ner_labels_gt])
#
# sample_model_name_or_path = 'distilbert-base-uncased'
# sample_tokenizer = DistilBertTokenizerFast.from_pretrained(sample_model_name_or_path)
#
# sample_encodings = sample_tokenizer(text=[w.text for w in sample_page.children.words],
#                                     truncation=True,
#                                     padding=True,
#                                     return_overflowing_tokens=True,
#                                     is_split_into_words=True)
