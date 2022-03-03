from commons.variables import paths
import os

from text_importation.classes import Commentary


# commentary_id, number,
for commentary_id in os.listdir(paths['base_dir']):
    commentary = Commentary(commentary_id)
