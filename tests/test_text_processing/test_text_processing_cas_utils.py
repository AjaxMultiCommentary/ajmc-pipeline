import shutil
from pathlib import Path

import pytest

from ajmc.commons import variables as vs
from ajmc.text_processing import cas_utils as casu
from tests import sample_objects as so


@pytest.mark.parametrize('ocr_commentary', [so.sample_ocrcommentary])
def test_export_commentary_to_xmis(ocr_commentary):
    base_xmi_dir = so.sample_comm_base_dir / vs.COMM_NER_ANN_REL_DIR
    output_xmi_dir = base_xmi_dir / so.sample_ocr_run_id / 'xmis'
    output_json_dir = output_xmi_dir.parent / 'jsons'

    # remove existing files before creating new output
    shutil.rmtree(output_xmi_dir, ignore_errors=True)

    casu.export_commentary_to_xmis(
            ocr_commentary,
            make_jsons=True,
            make_xmis=True,
            region_types=['commentary', 'app_crit'],
            xmis_dir=output_xmi_dir,
            jsons_dir=output_json_dir
    )

    # given that we're exporting to XMI appCrit and commentary
    # sections, it seems fair to assume a non-empty output of 
    # casu.export_commentary_to_xmis()
    assert len(list(output_xmi_dir.glob('*.xmi'))) > 0


def test_get_cas():
    test_xmi_path = Path(
        "tests/data/sample_commentaries/cu31924087948174/ner/annotation/xmi/1bm0b3_tess_final/cu31924087948174_0102.xmi")
    cas = casu.get_cas(test_xmi_path, vs.TYPESYSTEM_PATH)

    ajmc_metadata_type = 'webanno.custom.AjMCDocumentmetadata'
    metadata = cas.select(ajmc_metadata_type)[0]

    assert metadata.get('ocr_run_id') is not None
    assert metadata.get('region_types').split(',') is not None
    assert metadata.get('xmi_creation_date') is not None
