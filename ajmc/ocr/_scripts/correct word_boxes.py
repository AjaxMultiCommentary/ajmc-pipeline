import json

import cv2
from tqdm import tqdm

from ajmc.commons import variables as vs
from ajmc.ocr.pytorch.word_boxes_detection import get_word_boxes_by_dilation, get_word_boxes_by_projection, get_word_boxes_brute_force


for comm_id in vs.ALL_COMM_IDS:
    print('-----------------------------------')
    print(f'Processing {comm_id}')

    # We get the path to the images
    img_dir = vs.get_comm_img_dir(comm_id)
    ocr_output_dir = vs.get_comm_ocr_outputs_dir(comm_id, ocr_run_id='*_pytorch')

    # We get the path to an image file
    for img_path in tqdm(sorted(img_dir.glob('*.png'), key=lambda x: x.stem)[9:]):

        # We get the corresponding page json file
        ocr_output_path = ocr_output_dir / f'{img_path.stem}.json'

        # We open the page's json file
        page_dict = json.loads(ocr_output_path.read_text('utf-8'))

        # We open the pages image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        fallback_lines = 0
        for line in page_dict:
            line_img = img[line['xyxy'][1]:line['xyxy'][3] + 1, line['xyxy'][0]:line['xyxy'][2] + 1].copy()
            line_text = ' '.join([word['text'] for word in line['words']])
            word_boxes = get_word_boxes_by_dilation(line_img, line_text)
            if len(word_boxes) != len(line_text.split()):
                fallback_lines += 1
                try:
                    word_boxes = get_word_boxes_by_projection(line_img, line_text)
                except:
                    word_boxes = get_word_boxes_brute_force(line_img, line_text)

            word_boxes = [(w.xmin + line['xyxy'][0],
                           w.ymin + line['xyxy'][1],
                           w.xmax + line['xyxy'][0],
                           w.ymax + line['xyxy'][1]) for w in word_boxes]

            line['word_boxes'] = word_boxes
            line['words'] = [{'xyxy': w_box, 'text': w_text} for w_box, w_text in zip(word_boxes, line_text.split())]

        if fallback_lines > 0:
            print(f'Fallback for {img_path.stem}: {fallback_lines} lines')

        # Write the results to a file
        ocr_output_path.write_text(json.dumps(page_dict, indent=2))
