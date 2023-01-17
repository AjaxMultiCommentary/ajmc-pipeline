"""Use this script to bulk convert ocr runs to canonical commentary"""
from ajmc.commons.variables import PATHS, ALL_COMMENTARY_IDS
from ajmc.text_processing.canonical_classes import CanonicalCommentary
from ajmc.text_processing.ocr_classes import OcrCommentary
import os
from ajmc.commons.miscellaneous import stream_handler
from pathlib import Path
from tqdm import tqdm

stream_handler.setLevel(20)

DESIRED_COMMENTARIES = [
    # 'annalsoftacitusp00taci',
    # 'bsb10234118',
    # 'Colonna1975',
    # 'DeRomilly1976',
    # 'Ferrari1974',
    # 'Finglass2011',
    # 'Garvie1998',
    # 'Hermann1851',
    # 'Kamerbeek1953',
    # 'Paduano1982',
    # 'pvergiliusmaroa00virggoog',
    # 'Schneidewin_Nauck_Radermacher1913',
    'Stanford1963',
    # 'thukydides02thuc',
    # 'Untersteiner1934',
]


for comm_id in tqdm(DESIRED_COMMENTARIES, desc='Processing commentaries'):

    comm_dir = Path(PATHS['base_dir']) / comm_id
    runs_dir = comm_dir / 'ocr/runs'
    try:
        ocr_outputs_dir = next(runs_dir.glob('*_tess_base')) / 'outputs'
    except StopIteration:
        continue


    can = OcrCommentary.from_ajmc_structure(str(ocr_outputs_dir)).to_canonical()
    can.to_json(os.path.join(comm_dir, 'canonical/v2', ocr_outputs_dir.parent.stem + '.json'))
