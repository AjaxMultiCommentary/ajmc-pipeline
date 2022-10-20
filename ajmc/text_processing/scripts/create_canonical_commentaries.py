"""Use this script to bulk convert ocr runs to canonical commentary"""
from ajmc.commons.variables import PATHS
from ajmc.text_processing.canonical_classes import CanonicalCommentary
from ajmc.text_processing.ocr_classes import OcrCommentary
import os
from ajmc.commons.miscellaneous import stream_handler

stream_handler.setLevel(0)

comm_ids = [
    # {
    #     "id": "Colonna1975",
    #     "run": "28qlZa_tess_base",
    #     "split": "test"
    # },
    # {
    #     "id": "DeRomilly1976",
    #     "run": "28q0aU_tess_base",
    #     "split": "test"
    # },
    # {
    #     "id": "Ferrari1974",
    #     "run": "28r0zj_tess_base",
    #     "split": "test"
    # },
    # {
    #     "id": "Garvie1998",
    #     "run": "28r0KM_tess_base",
    #     "split": "test"
    # },
    # {
    #     "id": "Kamerbeek1953",
    #     "run": "28qloZ_tess_base",
    #     "split": "test"
    # },
    # {
    #     "id": "Paduano1982",
    #     "run": "28qlSF_tess_base",
    #     "split": "test"
    # },
    # {
    #     "id": "Untersteiner1934",
    #     "run": "28r0XU_tess_base",
    #     "split": "test"
    # },
    # {
    #     "id": "Wecklein1894",
    #     "run": "28r1pY_tess_base",
    #     "split": "test"
    # },
    # {
    #     "id": "bsb10234118",
    #     "run": "28qloR_tess_base",
    #     "split": "test"
    # },
    # {
    #     "id": "cu31924087948174",
    #     "run": "28qmab_tess_base",
    #     "split": "test"
    # },
    # {
    #     "id": "sophoclesplaysa05campgoog",
    #     "run": "15o09Y_lace_base_sophoclesplaysa05campgoog-2021-05-23-21-38-49-porson-2021-05-23-14-27-27",
    #     "split": "test"
    # },
    # {
    #     "id": "sophokle1v3soph",
    #     "run": "28r0iY_tess_base",
    #     "split": "test"
    # },
    {
        "id": "sophoclesplaysa05campgoog",
        "run": "28qm8n_tess_base",
        "split": "test"
    }
]

for comm in comm_ids:

    comm_dir = os.path.join(PATHS['base_dir'], comm['id'])
    runs_dir = os.path.join(comm_dir, 'ocr/runs')
    ocr_outputs_dir = os.path.join(runs_dir, comm['run'], 'outputs')
    can = OcrCommentary.from_ajmc_structure(ocr_outputs_dir).to_canonical()
    can.to_json(os.path.join(comm_dir, 'canonical/v2', comm['run'] + '.json'))
