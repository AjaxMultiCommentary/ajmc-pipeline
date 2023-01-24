"""Use this script to clean ocr/runs/xxxxxxx/outputs dirs from lace outputs and other weird files."""
# CHECKED 2023-01-23
import os

from ajmc.commons import variables as vs
from ajmc.commons.file_management.utils import walk_dirs

for comm_dir in walk_dirs(vs.COMMS_DATA_DIR):
    comm_runs_dir = vs.get_comm_ocr_runs_dir(comm_dir.name)
    if comm_runs_dir.is_dir():
        for ocr_run_dir in walk_dirs(comm_runs_dir):
            outputs_dir = vs.get_comm_ocr_outputs_dir(comm_dir.name, ocr_run_dir.name)
            for path in outputs_dir.glob('*'):
                if comm_dir.name not in path.name and path.suffix not in ['.sh', '']:
                    command = f'rm -rf {path}'
                    print(command)
                    os.system(command)
