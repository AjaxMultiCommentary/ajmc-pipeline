"""Use this script to binarize images of multiple commentary"""

import cv2

from ajmc.commons import variables
from ajmc.commons.image import binarize

comm_ids = [  # 'annalsoftacitusp00taci',
    'Finglass2011',
    # 'thukydides02thuc'
]

for comm_id in comm_ids:
    png_dir = variables.get_comm_img_dir(comm_id)
    jp2_dir = png_dir.parent / 'png_'

    for img_path in jp2_dir.glob('*.png'):
        img = cv2.imread(str(img_path))
        img = binarize(img)
        cv2.imwrite(str(png_dir / img_path.name), img)

#%%%
import cv2

from ajmc.commons.image import binarize
from pathlib import Path
dir_ = Path('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/Hermann1851/images/png')

for png in dir_.glob('*.png'):
    img = cv2.imread(str(png))
    img = binarize(img)
    cv2.imwrite(str(png), img)