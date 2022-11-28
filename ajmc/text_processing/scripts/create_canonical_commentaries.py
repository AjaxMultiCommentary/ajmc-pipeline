"""Use this script to bulk convert ocr runs to canonical commentary"""
from ajmc.commons.variables import PATHS, ALL_COMMENTARY_IDS
from ajmc.text_processing.canonical_classes import CanonicalCommentary
from ajmc.text_processing.ocr_classes import OcrCommentary
import os
from ajmc.commons.miscellaneous import stream_handler
from pathlib import Path
from tqdm import tqdm

stream_handler.setLevel(0)


for comm_id in tqdm(ALL_COMMENTARY_IDS, desc='Processing commentaries'):
    if comm_id.startswith('Colo') or comm_id.startswith('DeRo'):
        continue

    comm_dir = Path(PATHS['base_dir']) / comm_id
    runs_dir = comm_dir / 'ocr/runs'
    try:
        ocr_outputs_dir = next(runs_dir.glob('*_tess_base')) / 'outputs'
    except StopIteration:
        continue


    can = OcrCommentary.from_ajmc_structure(str(ocr_outputs_dir)).to_canonical()
    can.to_json(os.path.join(comm_dir, 'canonical/v2', ocr_outputs_dir.parent.stem + '.json'))
