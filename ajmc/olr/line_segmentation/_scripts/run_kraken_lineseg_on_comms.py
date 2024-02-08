import argparse
import json
from pathlib import Path

from kraken import pageseg, blla, binarization
from kraken.lib import vgsl
from PIL import Image
from tqdm import tqdm

from ajmc.commons.miscellaneous import ROOT_LOGGER

ROOT_LOGGER.setLevel('DEBUG')
COMMS_DIR = Path('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/')

parser = argparse.ArgumentParser()
parser.add_argument('--seg_type', type=str, default='legacy')
parser.add_argument('--model_path', type=str, default='/scratch/sven/anaconda3/envs/kraken/lib/python3.10/site-packages/kraken/blla.mlmodel')
parser.add_argument('--commentary_ids', nargs='+', type=str, default=['lestragdiesdeso00tourgoog'])

if __name__ == '__main__':
    args = parser.parse_args()

    if args.seg_type == 'blla':
        model = vgsl.TorchVGSLModel.load_model(args.model_path)

    assert args.seg_type in ['legacy', 'blla']

    for comm_dir in COMMS_DIR.iterdir():
        if comm_dir.is_dir() and (args.commentary_ids is None or comm_dir.stem in args.commentary_ids):
            output_dir = comm_dir / f'olr/lines/{args.seg_type}'
            output_dir.mkdir(parents=True, exist_ok=True)

            # Skip if already processed
            if len(list(output_dir.glob('*.json'))) == len(list((comm_dir / 'images/png').glob('*.png'))):
                print(f'Skipping {comm_dir.stem} as already processed')
                continue

            for img_path in tqdm(sorted((comm_dir / 'images/png').glob('*.png'), key=lambda x: x.stem), desc=f'Processing {comm_dir.stem}'):
                if img_path.stem.startswith('.'):
                    continue
                img = Image.open(img_path)
                output_path = output_dir / f'{img_path.stem}.json'
                try:
                    img = binarization.nlbin(img)
                    if args.seg_type == 'legacy':
                        output = pageseg.segment(img)
                    else:
                        output = blla.segment(img, model=model)
                except:
                    if args.seg_type == 'blla':
                        output = {'script_detection': False,
                                  'regions': [],
                                  'lines': [],
                                  'type': 'baselines',
                                  'text_direction': 'horizontal-lr'}
                    else:
                        output = {'boxes': [],
                                  "text_direction": "horizontal-lr",
                                  "script_detection": False}

                output_path.write_text(json.dumps(output, indent=2))
