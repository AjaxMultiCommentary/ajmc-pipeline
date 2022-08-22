from typing import List

from ajmc.commons.geometry import Shape


def read_yolo_txt_line(line: str,
                       ids_to_label,
                       image_width,
                       image_height
                       ):
    label_id, center_x, center_y, width, height, confidence = line.split(' ')
    label_id = int(label_id)
    label = ids_to_label[label_id]

    center_x = int(float(center_x) * image_width)
    center_y = int(float(center_y) * image_height)
    width = int(float(width) * image_width)
    height = int(float(height) * image_height)
    confidence = float(confidence)

    return {'label': label,
            'bbox': Shape.from_center_w_h(center_x, center_y, width, height),
            'conf': confidence}


def read_yolo_txt(lines: List[str],
                  ids_to_label,
                  image_width,
                  image_height
                  ):
    return [read_yolo_txt_line(line =l,
                               ids_to_label=ids_to_label,
                               image_width=image_width,
                               image_height=image_height) for l in lines]
