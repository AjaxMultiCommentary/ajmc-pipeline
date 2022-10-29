"""Use this script to export page jsons for the olr dataset"""
import json

from ajmc.commons.variables import PATHS, REGION_TYPES_TO_SEGMONTO, ROIS
from ajmc.text_processing.canonical_classes import CanonicalCommentary
import os
from ajmc.commons.miscellaneous import stream_handler

stream_handler.setLevel(0)

commentaries = [
    {
        "id": "annalsoftacitusp00taci",
        "run": "28o09e_tess_base",
    },
    {
        "id": "Wecklein1894",
        "run": "28r1pY_tess_base",
    },
    {
        "id": "bsb10234118",
        "run": "28qloR_tess_base",
    },
    {
        "id": "cu31924087948174",
        "run": "28qmab_tess_base",
    },
    {
        "id": "pvergiliusmaroa00virggoog",
        "run": "28o09d_tess_base",
    },
    {
        "id": "sophokle1v3soph",
        "run": "28r0iY_tess_base",
    },
    {
        "id": "sophoclesplaysa05campgoog",
        "run": "28qm8n_tess_base",
    },
    {
        "id": "thukydides02thuc",
        "run": "28o09a_tess_base",
    },

]

export_dir = '/Users/sven/Desktop/json_export'

for comm in commentaries:
    comm_export_dir = os.path.join(export_dir, comm['id'])
    os.makedirs(comm_export_dir, exist_ok=True)

    comm_dir = os.path.join(PATHS['base_dir'], comm['id'])
    can_path = os.path.join(comm_dir, PATHS['canonical'], comm['run'] + '.json')
    can_comm = CanonicalCommentary.from_json(can_path)

    for gt_page in can_comm.olr_groundtruth_pages:
        regions = [{'label': REGION_TYPES_TO_SEGMONTO[r.region_type],
                    'bbox': r.bbox.bbox}
                   for r in gt_page.children.regions
                   if r.region_type in ROIS]

        path = os.path.join(comm_export_dir, f'{gt_page.id}.json')
        with open(path, "w") as outfile:
            json.dump(regions, outfile, indent=4, ensure_ascii=False)
