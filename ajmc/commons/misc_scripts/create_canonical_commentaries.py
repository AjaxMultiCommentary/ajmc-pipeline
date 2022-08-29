"""Use this script to bulk convert ocr runs to canonical commentaries"""
from ajmc.commons.image import draw_page_regions_lines_words
from ajmc.commons.variables import PATHS
from ajmc.text_processing.canonical_classes import CanonicalCommentary
from ajmc.text_processing.ocr_classes import OcrCommentary
import os


comm_ids = [
    'annalsoftacitusp00taci',
    'pvergiliusmaroa00virggoog',
    'thukydides02thuc',
]

for comm_id in comm_ids:
    comm_dir = os.path.join(PATHS['base_dir'], comm_id)
    runs_dir = os.path.join(comm_dir, 'ocr/runs')
    run_name = [d for d in os.listdir(runs_dir) if d.endswith('tess_base')][0]
    ocr_outputs_dir = os.path.join(runs_dir, run_name, 'outputs')
    can = OcrCommentary.from_ajmc_structure(ocr_outputs_dir).to_canonical()
    can.to_json(os.path.join(comm_dir, 'canonical/v2', run_name+'.json'))
