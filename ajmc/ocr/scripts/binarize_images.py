"""Use this script to binarize images of multiple commentaries"""

import cv2
from ajmc.commons.variables import PATHS
import os
from ajmc.commons.image import binarize

comm_ids = [  # 'annalsoftacitusp00taci',
    'pvergiliusmaroa00virggoog',
    # 'thukydides02thuc'
]

for comm_id in comm_ids:
    jp2_dir = os.path.join(PATHS['base_dir'], comm_id, 'images/png_')
    png_dir = os.path.join(PATHS['base_dir'], comm_id, 'images/png')

    for img_name in os.listdir(jp2_dir):
        if img_name.endswith('.png'):
            img = cv2.imread(os.path.join(jp2_dir, img_name))
            img = binarize(img)
            cv2.imwrite(os.path.join(png_dir, img_name.replace('.png', '.png')),
                        img)
