from ajmc.text_processing import cas_utils
from ajmc.text_processing.ocr_classes import OcrCommentary
from pathlib import Path
from ajmc.commons import variables

comm_ids = [
    # 'annalsoftacitusp00taci',
    # 'bsb10234118',
    # 'Colonna1975',
    # 'DeRomilly1976',
    # 'Ferrari1974',
    # 'Finglass2011',
    # # 'Garvie1998',
    # 'Hermann1851',
    # # 'Kamerbeek1953',
    # # 'Paduano1982',
    # # 'pvergiliusmaroa00virggoog',
    # 'Schneidewin_Nauck_Radermacher1913',
    'Stanford1963',
    # 'thukydides02thuc',
    # 'Untersteiner1934',
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
