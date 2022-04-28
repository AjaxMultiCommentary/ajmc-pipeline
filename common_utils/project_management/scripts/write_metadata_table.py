from common_utils.variables import PATHS
import os

from text_importation.classes import Commentary


# commentary_id, number,
for commentary_id in os.listdir(PATHS['base_dir']):
    commentary = Commentary(commentary_id)
