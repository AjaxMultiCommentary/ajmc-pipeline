"""Use this script to binarize images of multiple commentary"""

import cv2
from ajmc.commons.variables import PATHS
import os
from ajmc.commons.image import binarize

comm_ids = [  # 'annalsoftacitusp00taci',
    'Finglass2011',
    # 'thukydides02thuc'
]

for comm_id in comm_ids:
    jp2_dir = os.path.join(PATHS['base_dir'], comm_id, 'images/png_')
    png_dir = os.path.join(PATHS['base_dir'], comm_id, 'images/png')

    for img_name in os.listdir(jp2_dir):
        if img_name.endswith('.png'):
            img = cv2.imread(os.path.join(jp2_dir, img_name))
            img = binarize(img)
            cv2.imwrite(os.path.join(png_dir, img_name.replace('.png', '.png')), img)

#%%%
import cv2

from ajmc.commons.image import binarize
from pathlib import Path
dir_ = Path('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/Hermann1851/images/png')

for png in dir_.glob('*.png'):
    img = cv2.imread(str(png))
    img = binarize(img)
    cv2.imwrite(str(png), img)