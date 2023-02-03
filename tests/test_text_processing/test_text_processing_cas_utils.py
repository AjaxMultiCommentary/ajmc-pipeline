import os
import ipdb
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
    vs.DRIVE_BASE_DIR = os.path.join(os.getcwd(), "tests", "data", "sample_commentaries")
    
    # just to be really sure that we don't delete production data
    # let's work on the sample data folder
    base_xmi_dir = Path(os.path.join(vs.DRIVE_BASE_DIR, ocr_commentary.id, vs.COMM_XMI_REL_DIR, 'xmi'))
    output_xmi_dir = next(base_xmi_dir.glob('*tess*'))
    ocr_run_id = os.path.basename(output_xmi_dir)
    output_json_dir = Path(os.path.join(vs.DRIVE_BASE_DIR, ocr_commentary.id, 'canonical', ocr_run_id))
    
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
    return