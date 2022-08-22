import json
import os
from collections import Counter

from transformers import LayoutLMv2TokenizerFast, LayoutLMv2ForTokenClassification

from ajmc.commons.geometry import is_rectangle_within_rectangle_with_threshold
from ajmc.commons.image import draw_rectangles
from ajmc.olr.layout_lm.config import create_olr_config
from ajmc.olr.layout_lm.layoutlm import get_data_dict_pages, align_predicted_page
from ajmc.olr.yolo.utils import read_yolo_txt

# Constants
BASE_DATA_DIR = '/mnt/ajmcdata1/drive_cached/AjaxMultiCommentary/data/commentaries/commentaries_data'
LAYOUTLM_XP_DIR = '/scratch/sven/layout_lm_tests/all_tokens_b'
YOLO_XP_DIR = '/scratch/sven/yolo/runs/binary_class'
CONFIGS_DIR = '/scratch/sven/yolo/configs'
MAP_PACKAGE_DIR = '/scratch/sven/packages/mAP'
WORD_INCLUSION_THRESHOLD = 0.5

# Read the experiments config
for config_name in sorted(next(os.walk(os.path.join(YOLO_XP_DIR, 'detect')))[1]):

    # Clean metrics dirs
    for command in [f"""rm -rf {os.path.join(MAP_PACKAGE_DIR, 'input/images-optional')}/*""",
                    f"""rm -rf {os.path.join(MAP_PACKAGE_DIR, 'input/ground-truth')}/*""",
                    f"""rm -rf {os.path.join(MAP_PACKAGE_DIR, 'input/detection-results')}/*"""]:
        os.system(command)

    # Get the config
    config_path = os.path.join(CONFIGS_DIR, config_name + '.json')
    config = create_olr_config(config_path, BASE_DATA_DIR)

    # Create the LayoutLM model and its tokenizer
    model_path = os.path.join(LAYOUTLM_XP_DIR, config_name, 'model')
    tokenizer = LayoutLMv2TokenizerFast.from_pretrained('microsoft/layoutlmv2-base-uncased')
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_path)

    # Get the pages
    pages = get_data_dict_pages(data_dict=config['data'])['eval']

    # Get the predictions
    for page in pages:
        words, labels = align_predicted_page(page=page,
                                             labels_to_ids=config['labels_to_ids'],
                                             ids_to_labels=config['ids_to_labels'],
                                             rois=config['rois'],
                                             regions_to_coarse_labels=config['region_types_to_labels'],
                                             tokenizer=tokenizer,
                                             model=model)

        for w, l in zip(words, labels):
            w.layout_lm_label = l

        # get YOLO's predictions
        yolo_preds_path = os.path.join(YOLO_XP_DIR, 'detect', config_name, 'labels')

        try :
            txt_name = [p for p in os.listdir(yolo_preds_path) if p.startswith(page.id)][0]
            txt_path = os.path.join(yolo_preds_path, txt_name)
            with open(txt_path, 'r') as f:
                lines = [l for l in f.readlines() if l]
            detected_regions = read_yolo_txt(lines=lines,
                                             # ids_to_label={0: 'region'},
                                             ids_to_label=config['ids_to_labels'],
                                             image_width=page.image.width,
                                             image_height=page.image.height)
        except IndexError:
            detected_regions = []

        # find the words in each region
        for r in detected_regions:
            r['words'] = [w.text for w in words
                          if is_rectangle_within_rectangle_with_threshold(contained=w.bbox.bbox,
                                                                          container=r['bbox'].bbox,
                                                                          threshold=WORD_INCLUSION_THRESHOLD)]
            r_labels = [w.layout_lm_label for w in words
                        if is_rectangle_within_rectangle_with_threshold(contained=w.bbox.bbox,
                                                                        container=r['bbox'].bbox,
                                                                        threshold=WORD_INCLUSION_THRESHOLD)]
            if r_labels:
                r['label'] = max(Counter(r_labels))
            else:
                r['label'] = 'no_words'

        # todo : filter empty regions
        # todo resize regions

        # Write image
        page.image.write(os.path.join(MAP_PACKAGE_DIR, 'input/images-optional', page.id + '.png'))

        # Write ground-truth
        lines = []
        for r in page.children['region']:
            if r.info['region_type'] in config['rois']:
                line = [region_types_to_labels[r.info['region_type']]]
                line += [str(el) for el in r.bbox.xyxy]
                lines.append(' '.join(line))

        with open(os.path.join(MAP_PACKAGE_DIR, 'input/ground-truth', page.id + '.txt'), 'w') as f:
            f.write('\n'.join(lines))

        # Write prediction
        lines = []
        for r in detected_regions:
            line = [r['label'],str(r['conf'])]
            line += [str(el) for el in r['bbox'].xyxy]
            lines.append(' '.join(line))

        with open(os.path.join(MAP_PACKAGE_DIR, 'input/detection-results', page.id + '.txt'), 'w') as f:
            f.write('\n'.join(lines))

        # # Draw yolo prediction
        # matrix = draw_rectangles(rectangles=[r['bbox'].bbox for r in detected_regions],
        #                          matrix=page.image.matrix.copy(),
        #                          output_path=os.path.join(YOLO_XP_DIR, 'detect', config_name, page.id+'.png'))


    # Run evaluation
    print(f'**** Evaluating {config_name}')
    for command in [f"""cd {MAP_PACKAGE_DIR}; python main.py -na -np""",
                    f"""mv {os.path.join(MAP_PACKAGE_DIR, 'output/output.txt')} {os.path.join(YOLO_XP_DIR, 'train', config_name, 'Cartucho_mAP_results.txt')}""",
                    ]:
        os.system(command)


