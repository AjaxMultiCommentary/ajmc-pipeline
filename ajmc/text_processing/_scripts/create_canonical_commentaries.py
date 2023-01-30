"""Use this script to bulk convert ocr runs to canonical commentary"""
from tqdm import tqdm

from ajmc.commons import variables
from ajmc.commons.miscellaneous import stream_handler
from ajmc.text_processing.ocr_classes import OcrCommentary

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

    try:
        ocr_outputs_dir = next(variables.get_comm_ocr_runs_dir(comm_id).glob('*_tess_base')) / 'outputs'
    except StopIteration:
        continue

    OcrCommentary.from_ajmc_data(id=comm_id).to_canonical().to_json()
