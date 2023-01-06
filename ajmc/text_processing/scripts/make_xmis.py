from ajmc.text_processing import cas_utils
from ajmc.text_processing.ocr_classes import OcrCommentary
from pathlib import Path
from ajmc.commons import variables

comm_ids = [
    'Colonna1975',
    # 'pvergiliusmaroa00virggoog',
]

skiped = []
for comm_id in comm_ids:
    runs_dir = Path(variables.PATHS['base_dir']) / comm_id / variables.PATHS['ocr']
    try:
        ocr_dir = next(runs_dir.glob('*tess_base'))
    except StopIteration:
        skiped.append(comm_id)
        continue

    ocr_outputs_dir = ocr_dir / 'outputs'
    comm = OcrCommentary.from_ajmc_structure(str(ocr_outputs_dir))
    cas_utils.export_commentary_to_xmis(comm, make_jsons=True, make_xmis=True, region_types=['app_crit'])
