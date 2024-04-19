import json
from pathlib import Path

import cv2
import torch
from torchvision.io import read_image, ImageReadMode

from ajmc.commons import variables as vs, geometry as geom
from ajmc.commons.file_management import get_62_based_datecode
from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.ocr.pytorch import data_processing as dp
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.model import OcrTorchModel
from ajmc.ocr.pytorch.word_boxes_detection import get_word_boxes_by_projection, get_word_boxes_by_dilation, get_word_boxes_brute_force

logger = get_ajmc_logger(__name__)
MODEL_DIR = Path('/scratch/sven/withbackbone_v2')

config = get_config(MODEL_DIR / '1A_withbackbone_new.json')
model = OcrTorchModel(config)
model_snapshot = torch.load(MODEL_DIR / 'best_model.pt')
model.load_state_dict(model_snapshot['MODEL_STATE'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device {device}')
model.to(device)

max_batch_size = 32


def get_page_lines(img_path: Path):
    img_tensor = read_image(str(img_path), mode=ImageReadMode.GRAY).requires_grad_(False)

    page_id = img_path.stem
    comm_id = img_path.stem.split('_')[0]

    line_detection_preds_path = vs.get_comm_olr_lines_dir(comm_id) / 'combined' / f'{page_id}.json'
    line_shapes = [geom.Shape(l) for l in json.loads(line_detection_preds_path.read_text())]
    line_tensors = [img_tensor[:, l.ymin:l.ymax + 1, l.xmin:l.xmax + 1].clone() for l in line_shapes]
    return [{'shape': l_shape,
             'torch_line': dp.TorchInferenceLine(img_tensor=l_tensor,
                                                 img_height=config['chunk_height'],
                                                 chunk_width=config['chunk_width'],
                                                 chunk_overlap=config['chunk_overlap'])}
            for l_shape, l_tensor in zip(line_shapes, line_tensors)]


for comm_id in vs.ALL_COMM_IDS:
    output_dir = vs.get_comm_ocr_runs_dir(comm_id) / f'{get_62_based_datecode()}_pytorch' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(vs.get_comm_img_dir(comm_id).glob('*' + vs.DEFAULT_IMG_EXTENSION), key=lambda p: p.stem):
        logger.info(f'Processing {img_path.stem}')
        lines = get_page_lines(img_path)

        # Run the model on each page, dividing it into batches
        page_texts = []
        for batch in dp.batch_builder(ocr_lines=[d['torch_line'] for d in lines], max_batch_size=max_batch_size, batch_class=dp.TorchInferenceBatch):
            line_texts = model.predict(batch.chunks.to(device), batch.chunks_to_img_mapping, batch.img_widths, remove_repetitions=True)
            page_texts.extend(line_texts)

        # Store the strings in
        for j, line_text in enumerate(page_texts):
            lines[j]['text'] = line_text

        # Get the words boxes
        cv2_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        fallback_lines = 0
        for j, line in enumerate(lines):
            if line['text'] == '':
                line['word_boxes'] = []
                continue
            cv2_line = cv2_img[line['shape'].ymin:line['shape'].ymax + 1, line['shape'].xmin:line['shape'].xmax + 1].copy()
            word_boxes = get_word_boxes_by_dilation(cv2_line, line['text'])
            if len(word_boxes) != len(line['text'].split()):
                fallback_lines += 1
                try:
                    word_boxes = get_word_boxes_by_projection(cv2_line, line['text'])
                except:
                    word_boxes = get_word_boxes_brute_force(cv2_line, line['text'])

            word_boxes = [(w.xmin + line['shape'].xmin,
                           w.ymin + line['shape'].ymin,
                           w.xmax + line['shape'].xmin,
                           w.ymax + line['shape'].ymin) for w in word_boxes]

            line['word_boxes'] = word_boxes

        if fallback_lines > 0:
            logger.warning(f'Fallback for {img_path.stem}: {fallback_lines} lines')

        # Write the results to a file
        output = [{'xyxy': line['shape'].xyxy,
                   'words': [{'xyxy': w_box, 'text': w_text}
                             for w_box, w_text in zip(line['word_boxes'], line['text'].split())]}
                  for line in lines]

        output_path = output_dir / f'{img_path.stem}.json'
        output_path.write_text(json.dumps(output, indent=2))
