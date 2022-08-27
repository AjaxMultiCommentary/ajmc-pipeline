import os
from ajmc.commons.file_management.utils import get_62_based_datecode
from ajmc.text_processing.ocr_classes import OcrCommentary
from ajmc.commons.variables import PATHS

TESSDATA_MAP = {
    "bsb10234118": "lat+grc+GT4HistOCR_50000000.997_191951",
    "Colonna1975": "lat+grc+GT4HistOCR_50000000.997_191951",
    "cu31924087948174": "eng+grc+GT4HistOCR_50000000.997_191951",
    "Ferrari1974": "ita+grc+GT4HistOCR_50000000.997_191951",
    "Garvie1998": "eng+grc+GT4HistOCR_50000000.997_191951",
    "Kamerbeek1953": "eng+grc+GT4HistOCR_50000000.997_191951",
    "Paduano1982": "ita+grc+GT4HistOCR_50000000.997_191951",
    "sophoclesplaysa05campgoog": "eng+grc+GT4HistOCR_50000000.997_191951",
    "sophokle1v3soph": "deu+grc+GT4HistOCR_50000000.997_191951",
    "Untersteiner1934": "ita+grc+GT4HistOCR_50000000.997_191951",
    "Wecklein1894": "deu+frk+grc+GT4HistOCR_50000000.997_191951",
}

for comm_id, langs in TESSDATA_MAP.items():
    comm_dir = os.path.join(PATHS['base_dir'], comm_id)
    png_dir = os.path.join(comm_dir, PATHS['png'])

    runs_dir = os.path.join(comm_dir, 'ocr/runs/')
    run_name = [d for d in os.listdir(runs_dir) if d.endswith('_tess_base')][0]
    output_dir =os.path.join(runs_dir, run_name, 'outputs')

    command = f"""mv {os.path.join(png_dir, '*.hocr')} {output_dir}"""
    os.system(command)
    print(comm_id, '     ', run_name)

    can = OcrCommentary.from_ajmc_structure(output_dir).to_canonical()
    can.to_json(os.path.join(PATHS['base_dir'], comm_id, PATHS['canonical'], run_name+'.json'))
