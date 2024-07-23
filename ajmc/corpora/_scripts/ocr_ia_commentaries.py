from pathlib import Path
import json
import json
from pathlib import Path

import torch
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

from ajmc.commons import geometry as geom
from ajmc.commons.miscellaneous import ROOT_LOGGER
from ajmc.ocr.pytorch import data_processing as dp
from ajmc.ocr.pytorch.config import get_config
from ajmc.ocr.pytorch.model import OcrTorchModel

ROOT_LOGGER.setLevel('WARNING')

root_dir = Path('/mnt/ajmcdata1/data/ia_commentaries/data')

ocr_model_dir = Path('/scratch/sven/withbackbone_v2')

config = get_config(ocr_model_dir / '1A_withbackbone_new.json')
model = OcrTorchModel(config)
model_snapshot = torch.load(ocr_model_dir / 'best_model.pt')
model.load_state_dict(model_snapshot['MODEL_STATE'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')
model.to(device)

max_batch_size = 32

for zip_path in tqdm(sorted(root_dir.glob('*.zip')), desc='Processing zip files'):
    img_extension = zip_path.stem[-3:]

    # Create a directory for each zip file
    comm_dir = root_dir / zip_path.stem
    comm_dir.mkdir(exist_ok=True)

    # Create the binarized images directory
    binarized_dir = comm_dir / f'{zip_path.stem}_binarized_png'

    # Create the ocr directory
    ocr_dir = comm_dir / 'ocr'
    ocr_dir.mkdir(exist_ok=True)

    # Process the images
    for img_path in tqdm(sorted(binarized_dir.glob(f'*.png'))):
        # Binarize the image and convert it to png
        img_tensor = read_image(str(img_path), mode=ImageReadMode.GRAY).requires_grad_(False)
        line_shapes = json.load((comm_dir / f'lines/{img_path.stem}.json').open('r', encoding='utf-8'))
        line_shapes = [geom.Shape.from_xyxy(*l) for l in line_shapes]
        line_tensors = [img_tensor[:, l.ymin:l.ymax + 1, l.xmin:l.xmax + 1].clone() for l in line_shapes]

        lines = [{'shape': l_shape,
                  'torch_line': dp.TorchInferenceLine(img_tensor=l_tensor,
                                                      img_height=config['chunk_height'],
                                                      chunk_width=config['chunk_width'],
                                                      chunk_overlap=config['chunk_overlap'])}
                 for l_shape, l_tensor in zip(line_shapes, line_tensors)]

        # Run the model on each page, dividing it into batches
        page_texts = []
        for batch in dp.batch_builder(ocr_lines=[d['torch_line'] for d in lines], max_batch_size=max_batch_size, batch_class=dp.TorchInferenceBatch):
            line_texts = model.predict(batch.chunks.to(device), batch.chunks_to_img_mapping, batch.img_widths, remove_repetitions=True)
            page_texts.extend(line_texts)

        # Store the strings in
        for j, line_text in enumerate(page_texts):
            lines[j]['text'] = line_text

        # Write the results to a file
        output = [{'xyxy': line['shape'].xyxy, 'text': line['text']} for line in lines]
        with open(ocr_dir / f'{img_path.stem}.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
