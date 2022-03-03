import os
import random

from commons.variables import paths
import shutil


def move_files_in_each_commentary_dir(src: str, dst: str):
    """Moves/rename files/folders in the folder structur

    Args:
        src: relative path of the source file/folder, from commentary root (e.g. `'ocr/groundtruth'`)
        dst: relative path of the destination file/folder,  from commentary root (e.g. `'ocr/groundtruth'`)
    """

    for dir_name in next(os.walk(paths['base_dir']))[1]:
        if os.path.exists(os.path.join(paths['base_dir'], dir_name, src)):
            # Moves the file/folder
            shutil.move(os.path.join(paths['base_dir'], dir_name, src),
                        os.path.join(paths['base_dir'], dir_name, dst))



def create_folder_in_each_commentary_dir(rel_path:str):
    for dir_name in next(os.walk(paths['base_dir']))[1]:
        os.makedirs(os.path.join(paths['base_dir'], dir_name, rel_path), exist_ok=True)


move_files_in_each_commentary_dir('ocr/ocr_general_evaluation', 'ocr/general_evaluation')
create_folder_in_each_commentary_dir('ner/annotation/xmi')

#%% encode ocr runs

from datetime import datetime
from utils.project_management.utils import get_62_based_datecode, int_to_62_based_code

codes = []
for dir_name in os.listdir(paths['base_dir']):

    if os.path.exists(os.path.join(paths['base_dir'], dir_name, 'ocr/runs')):
        runs_path = os.path.join(paths['base_dir'], dir_name, 'ocr/runs')

        for ocr_run in os.listdir(runs_path):

            cdate = datetime.utcfromtimestamp(os.stat(os.path.join(runs_path, ocr_run)).st_birthtime)
            code = get_62_based_datecode(cdate)
            while code in codes:
                code = code[0:3]+int_to_62_based_code(random.randint(0, 86399))

            shutil.move(os.path.join(runs_path, ocr_run), os.path.join(runs_path, code+'_'+ocr_run[7:]))
            codes.append(code)


