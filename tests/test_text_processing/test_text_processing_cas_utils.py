import os
import pytest
from pathlib import Path
from tests import sample_objects as so
from ajmc.commons import variables as vs
from ajmc.text_processing import cas_utils as casu

def clean_xmi_directory(path : Path) -> None:
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        os.remove(filepath)
        print(f'removed file {filepath}')

@pytest.mark.parametrize('ocr_commentary', [so.sample_ocrcommentary])
def test_export_commentary_to_xmis(ocr_commentary):
    # work only on the test data not on the data in the Drive
    comm_base_dir = so.sample_comm_base_dir
    
    # just to be really sure that we don't delete production data
    # let's work on the sample data folder
    base_xmi_dir = comm_base_dir / vs.COMM_XMI_REL_DIR
    output_xmi_dir = next(base_xmi_dir.glob('*tess*'))
    ocr_run_id = os.path.basename(output_xmi_dir)
    output_json_dir = comm_base_dir / 'canonical' / ocr_run_id
    
    # remove existing files before creating new output
    clean_xmi_directory(output_xmi_dir)

    casu.export_commentary_to_xmis(
        ocr_commentary,
        make_jsons=True,
        make_xmis=True,
        region_types=['commentary','app_crit'],
        xmi_dir=output_xmi_dir,
        json_dir=output_json_dir
    )

    # given that we're exporting to XMI appCrit and commentary
    # sections, it seems fair to assume a non-empty output of 
    # casu.export_commentary_to_xmis()
    assert len(os.listdir(output_xmi_dir)) > 0

def test_get_cas():
    test_xmi_path = "tests/data/sample_commentaries/cu31924087948174/ner/annotation/xmi/1bm0b3_tess_final/cu31924087948174_0102.xmi"
    cas = casu.get_cas(test_xmi_path, vs.TYPESYSTEM_PATH)
    
    ajmc_metadata_type = 'webanno.custom.AjMCDocumentmetadata'
    metadata = cas.select(ajmc_metadata_type)[0]

    assert metadata.get('ocr_run_id') is not None
    assert metadata.get('regions_considered').split(',') is not None
    assert metadata.get('xmi_creation_date') is not None