"""⚙️ WIP module to pagify an OCRdataset"""
from pathlib import Path

import cv2
import numpy as np

from ajmc.commons import variables as vs
from ajmc.ocr import variables as ocr_vs


def get_commentary_image_dimension(comm_id: str):
    img_dir = vs.get_comm_img_dir(comm_id)
    img_path = [p for p in img_dir.glob(f'*{vs.DEFAULT_IMG_EXTENSION}')][10]
    return cv2.imread(str(img_path)).shape


def create_blank_image(shape, color=(255, 255, 255)):
    image = np.zeros(shape, np.uint8)
    image[:] = color
    return image


def insert_line(background, line, x, y):
    h, w = line.shape[:2]
    background[y:y + h, x:x + w] = line
    return background


def insert_lines(background, lines, margin=0.1, interline=1.5):
    x = int(margin * background.shape[1])
    y = int(margin * background.shape[0])

    for line in lines:
        background = insert_line(background, line, x, y)
        y += int(interline * line.shape[0])

    return background


def pagify_dataset(dataset_dir: Path,
                   output_dir: Path,
                   margin=0.1,
                   interline=1.5):
    previous_comm_id = None
    page_img_lines = []
    page_txt_lines = []
    page_number = 0

    for img_path in dataset_dir.glob(f'*{ocr_vs.IMG_EXTENSION}'):
        page_img_lines.append(cv2.imread(str(img_path)))
        page_txt_lines.append(img_path.with_suffix('.gt.txt').read_text(encoding='utf-8'))
        comm_id = img_path.name.split('_')[0]

        if len(page_img_lines) > 0 and \
                previous_comm_id is not None and \
                (len(page_img_lines) == 20 or comm_id != previous_comm_id):
            page_img = insert_lines(create_blank_image(get_commentary_image_dimension(previous_comm_id)),
                                    page_img_lines)

            # Write the resulting image and text
            new_img_path = output_dir / f'{previous_comm_id}_{page_number}{ocr_vs.IMG_EXTENSION}'
            cv2.imwrite(str(new_img_path), page_img)
            new_img_path.with_suffix('.gt.txt').write_text('\n'.join(page_txt_lines), encoding='utf-8')

            page_img_lines = []
            page_txt_lines = []
            page_number += 1

        previous_comm_id = comm_id
