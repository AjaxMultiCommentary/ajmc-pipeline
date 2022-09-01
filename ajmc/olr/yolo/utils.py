from typing import List

from ajmc.commons.geometry import Shape


def parse_yolo_txt_line(line: str,
                        ids_to_label,
                        image_width,
                        image_height,
                        is_groundtruth: bool
                        ):
    if is_groundtruth:
        label_id, center_x, center_y, width, height = line.split(' ')
        confidence = None
    else:
        label_id, center_x, center_y, width, height, confidence = line.split(' ')
        confidence = float(confidence)

    label_id = int(label_id)
    label = ids_to_label[label_id]
    center_x = int(float(center_x) * image_width)
    center_y = int(float(center_y) * image_height)
    width = int(float(width) * image_width)
    height = int(float(height) * image_height)

    return {'label': label,
            'label_id': label_id,
            'bbox': Shape.from_center_w_h(center_x, center_y, width, height),
            'conf': confidence}


def parse_yolo_txt(path: str,
                   ids_to_label,
                   image_width,
                   image_height,
                   is_groundtruth:bool
                   )-> List[dict]:

    with open(path, 'r') as f:
        lines = [l for l in f.readlines() if l]

    return [parse_yolo_txt_line(line =l,
                                ids_to_label=ids_to_label,
                                image_width=image_width,
                                image_height=image_height,
                                is_groundtruth=is_groundtruth) for l in lines]
