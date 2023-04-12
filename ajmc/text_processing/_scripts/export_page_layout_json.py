"""Use this script to export page jsons for the olr dataset"""
import json
from pathlib import Path

from ajmc.commons import variables
from ajmc.commons.miscellaneous import stream_handler
from ajmc.text_processing.canonical_classes import CanonicalCommentary

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

export_dir = Path('/Users/sven/Desktop/json_export')

for comm in commentaries:
    comm_export_dir = export_dir / comm['id']
    comm_export_dir.mkdir(exist_ok=True, parents=True)
    can_path = variables.get_comm_ner_jsons_dir(comm['id']) / (comm['run'] + '.json')
    can_comm = CanonicalCommentary.from_json(can_path)

    for gt_page in can_comm.olr_gt_pages:
        regions = [{'label': variables.REGION_TYPES_TO_SEGMONTO[r.region_type],
                    'bbox': r.bbox.bbox}
                   for r in gt_page.children.regions
                   if r.region_type in variables.ROIS]

        (comm_export_dir / f'{gt_page.id}.json').write_text(
            json.dumps(regions, indent=2, ensure_ascii=False, encoding='utf-8'))
