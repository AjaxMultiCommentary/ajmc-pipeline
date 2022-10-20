import os
import cv2
from ajmc.commons.file_management.utils import walk_files
from ajmc.commons.variables import PATHS
import numpy as np


def get_commentary_image_dimension(comm_id):
    png_path = os.path.join(PATHS['base_dir'], comm_id, PATHS['png'])
    img_path = [p for p in walk_files(png_path, filter=lambda x: x.suffix == '.png')][10]
    img = cv2.imread(str(img_path))
    return img.shape


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


def pagify_dataset(dataset_dir, output_dir, margin=0.1, interline=1.5):
    previous_comm_id = None
    page_img_lines = []
    page_txt_lines = []
    page_number = 0

    for img_path in walk_files(dataset_dir, filter=lambda x: x.suffix == '.png'):
        page_img_lines.append(cv2.imread(str(img_path)))
        page_txt_lines.append(img_path.with_suffix('.gt.txt').read_text())
        comm_id = img_path.name.split('_')[0]

        if len(page_img_lines) > 0 and \
                previous_comm_id is not None and \
                (len(page_img_lines) == 20 or comm_id != previous_comm_id):
            page_img = insert_lines(create_blank_image(get_commentary_image_dimension(previous_comm_id)),
                                    page_img_lines)
            cv2.imwrite(os.path.join(output_dir, f'{previous_comm_id}_{page_number}.png'), page_img)
            with open(os.path.join(output_dir, f'{previous_comm_id}_{page_number}.gt.txt'), 'w') as f:
                f.write('\n'.join(page_txt_lines))

            page_img_lines = []
            page_txt_lines = []
            page_number += 1

        previous_comm_id = comm_id

