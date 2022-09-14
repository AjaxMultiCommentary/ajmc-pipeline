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

    name_len = len([fname for fname in os.listdir(png_dir) if fname.endswith('.png')][0].split('.')[0])
    run_name = get_62_based_datecode()+'_tess_base'

    run_dir = os.path.join(comm_dir, 'ocr/runs/', run_name)
    output_dir =os.path.join(run_dir, 'outputs')
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    command = f"""cd {png_dir}; export TESSDATA_PREFIX=/Users/sven/packages/tesseract/tessdata/; for i in *.png ; do tesseract $i "${{i:0:{name_len}}}" -l {langs} /Users/sven/packages/tesseract/tess_config;  done;"""
    with open(os.path.join(run_dir,'command.sh'), 'w') as f:
        f.write(command)
    os.system(command=command)

    can = OcrCommentary.from_ajmc_structure(output_dir).to_canonical()
    can.to_json(os.path.join(PATHS['base_dir'], comm_id, PATHS['canonical'], run_name+'.json'))
