import os
from ajmc.commons.variables import PATHS
from ajmc.commons import get_62_based_datecode


def get_kraken_command(commentary_id, model_path):
    model_name = model_path.split('/')[-1].split('.')[0]
    
    ocr_dir = get_62_based_datecode()+'_'+model_name
    ocr_path = os.path.join(PATHS['base_dir'], commentary_id, 'ocr/runs/' + ocr_dir)
    os.makedirs(ocr_path, exist_ok=True)
    
    png_abs_path = os.path.join(PATHS['base_dir'], commentary_id, PATHS['png'])
    image_names = sorted([fname for fname in os.listdir(png_abs_path) if fname.endswith('.png')])
    image_paths = [os.path.join(png_abs_path, f) for f in image_names]
    ocr_paths = [os.path.join(ocr_path, f[:-3] + 'hocr') for f in image_names]
    
    file_list = ' '.join([f'-i {img} {ocr}' for img, ocr in zip(image_paths, ocr_paths)])
    command = ' '.join(['kraken', file_list, '-h segment ocr --model '+model_path])
    return command


command=get_kraken_command('sophokle1v3soph',
                   '/home/najem/packages/kraken/OCR-kraken-models/kraken-models/greek-german_serifs_sophokle1v3soph/greek-german_serifs_sophokle1v3soph.mlmodel')


