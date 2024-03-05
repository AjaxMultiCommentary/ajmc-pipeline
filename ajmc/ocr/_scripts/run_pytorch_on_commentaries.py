# PROCESSING FOR A SINGLE IMAGE

# Load the image as torch tensor (H, W, C)
# Load the line detection predictions
# Slice the tensors, creating to new tensors (H, W, C) for each line
# Run the model on each line
# Save the results as a json file

import json
from pathlib import Path

import cv2
import torch
from torchvision.io import read_image, ImageReadMode

from ajmc.commons import variables as vs, geometry as geom
from ajmc.ocr.pytorch import data_processing as dp
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.model import OcrTorchModel

comm_id = 'sophoclesplaysa05campgoog'
page_id = 'sophoclesplaysa05campgoog_0146'

MODEL_DIR = Path('/scratch/sven/withbackbone_v2')
DATA_DIR = Path('/scratch/sven/ajmc_data/commentaries_data/sophoclesplaysa05campgoog/images/png/')

config = get_config(MODEL_DIR / '1A_withbackbone_new.json')
model = OcrTorchModel(config)
model_snapshot = torch.load(MODEL_DIR / 'best_model.pt')
model.load_state_dict(model_snapshot['MODEL_STATE'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

max_batch_size = 32


def get_lines(img_path: Path):
    img_tensor = read_image(str(img_path), mode=ImageReadMode.GRAY).requires_grad_(False)

    page_id = img_path.stem
    comm_id = img_path.stem.split('_')[0]

    line_detection_preds_path = vs.get_comm_olr_lines_dir(comm_id) / 'combined' / f'{page_id}.json'

    line_detection_preds = [geom.Shape(l) for l in json.loads(line_detection_preds_path.read_text())]

    lines = {l: img_tensor[l.ymin:l.ymax, l.xmin:l.xmax].clone()
             for l in line_detection_preds}

    lines = {l: dp.TorchInferenceLine(line_id='test', img_tensor=v, img_height=config['chunk_height'], chunk_width=config['chunk_width'])
             for l, v in lines.items()}
    return lines


img_path = DATA_DIR / 'sophoclesplaysa05campgoog_0146.png'
img_tensor = read_image(str(img_path), mode=ImageReadMode.GRAY).requires_grad_(False)

img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

page_id = img_path.stem
comm_id = img_path.stem.split('_')[0]

line_detection_preds_path = vs.get_comm_olr_lines_dir(comm_id) / 'combined' / f'{page_id}.json'

line_shapes = [geom.Shape(l) for l in json.loads(line_detection_preds_path.read_text())]

cv2_lines = [img[l.ymin:l.ymax + 1, l.xmin:l.xmax + 1].copy() for l in line_shapes]

line_tensors = [img_tensor[:, l.ymin:l.ymax + 1, l.xmin:l.xmax + 1].clone()
                for l in line_shapes]

line_redimensionning_factors = [l.shape[1] / config['chunk_height'] for l in line_tensors]

torch_lines = [dp.TorchInferenceLine(line_id='test', img_tensor=v, img_height=config['chunk_height'], chunk_width=config['chunk_width'],
                                     chunk_overlap=config['chunk_overlap']) for v in line_tensors]

page_strings = []
for batch in dp.batch_builder(ocr_lines=torch_lines, max_batch_size=max_batch_size, batch_class=dp.TorchInferenceBatch):
    # Run the model
    strings, offsets = model.predict(batch.chunks.to(device), batch.chunks_to_img_mapping, batch.img_widths, remove_repetitions=True)
    page_strings.extend(strings)

for i, (line_img, line_string) in enumerate(zip(cv2_lines, page_strings)):
    word_boxes = get_word_boxes(line_img, line_string, draw_path=f'/scratch/sven/{page_id}_{i}.png')

#%% Try with opencv
