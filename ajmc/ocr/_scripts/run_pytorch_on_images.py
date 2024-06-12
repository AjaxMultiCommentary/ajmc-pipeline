"""This is a quick notebook to run the in-house torch model on images or pdfs."""

from pathlib import Path

from ajmc.olr.line_detection.models import KrakenLegacyModel

data_root_dir = Path('/scratch/sven/Bultmann_Theol_test_2023_06_13/')
# data_root_dir = Path('/Users/sven/Desktop/Bultmann_Theol_test_2023_06_13/')

imgs_dir = data_root_dir / 'images'

# %% Make sure the images are single channel
from PIL import Image
import numpy as np

for img_path in imgs_dir.glob('*.png'):
    img = Image.open(img_path)
    img = img.convert('L')
    img = np.array(img)
    img = Image.fromarray(img)
    img.save(img_path)

# %% Run the model on the images
import json
from pathlib import Path

import torch
from torchvision.io import read_image, ImageReadMode

from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.ocr.pytorch import data_processing as dp
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.model import OcrTorchModel

logger = get_ajmc_logger(__name__)
MODEL_DIR = Path('/scratch/sven/withbackbone_v2')

config = get_config(MODEL_DIR / '1A_withbackbone_new.json')
model = OcrTorchModel(config)
model_snapshot = torch.load(MODEL_DIR / 'best_model.pt')
model.load_state_dict(model_snapshot['MODEL_STATE'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
logger.info(f'Using device {device}')
model.to(device)

line_detection_model = KrakenLegacyModel()

output_dir = data_root_dir / 'ocr_output'
output_dir.mkdir(exist_ok=True)

for img_path in sorted(imgs_dir.glob('*.png'), key=lambda p: p.stem):
    logger.info(f'Processing {img_path.stem}')

    line_shapes = line_detection_model.predict(img_path)
    img_tensor = read_image(str(img_path), mode=ImageReadMode.GRAY).requires_grad_(False)

    line_tensors = [img_tensor[:, l.ymin:l.ymax + 1, l.xmin:l.xmax + 1].clone() for l in line_shapes]

    lines = [{'shape': l_shape,
              'torch_line': dp.TorchInferenceLine(img_tensor=l_tensor,
                                                  img_height=config['chunk_height'],
                                                  chunk_width=config['chunk_width'],
                                                  chunk_overlap=config['chunk_overlap'])}
             for l_shape, l_tensor in zip(line_shapes, line_tensors)]

    page_texts = []
    for batch in dp.batch_builder(ocr_lines=[d['torch_line'] for d in lines], max_batch_size=8, batch_class=dp.TorchInferenceBatch):
        line_texts = model.predict(batch.chunks.to(device), batch.chunks_to_img_mapping, batch.img_widths, remove_repetitions=True)
        page_texts.extend(line_texts)

    for line, text in zip(lines, page_texts):
        line['text'] = text

    with open(output_dir / f'{img_path.stem}.json', 'w', encoding='utf-8') as f:
        json.dump([{'shape': l['shape'].xyxy, 'text': l['text']} for l in lines], f, ensure_ascii=False, indent=4)

    # Also export a .txt
    (output_dir / f'{img_path.stem}.txt').write_text('\n'.join([l['text'] for l in lines]), encoding='utf-8')
