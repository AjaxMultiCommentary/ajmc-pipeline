import json
import shutil
import zipfile
from pathlib import Path

import cv2
from tqdm import tqdm

from ajmc.commons.miscellaneous import get_ajmc_logger, ROOT_LOGGER
from ajmc.olr.line_detection import models as LDModels

ROOT_LOGGER.setLevel('WARNING')

root_dir = Path('/mnt/ajmcdata1/data/ia_commentaries/data')

line_detection_model = LDModels.CombinedModel(
    adjusted_legacy_model=LDModels.AdjustedModel(LDModels.KrakenLegacyModel()),
    adjusted_blla_model=LDModels.AdjustedModel(
        LDModels.KrakenBllaModel(model_path=Path('/scratch/sven/anaconda3/envs/kraken/lib/python3.10/site-packages/kraken/blla.mlmodel'))),
)

for zip_path in tqdm(sorted(root_dir.glob('*.zip'))[:50], desc='Processing zip files'):
    img_extension = zip_path.stem[-3:]

    # Create a directory for each zip files
    comm_dir = root_dir / zip_path.stem
    comm_dir.mkdir(exist_ok=True)
    img_dir = comm_dir / zip_path.stem  # Doesn't need to be created as it will be created by the unzip function

    # Create the binarized images directory
    binarized_dir = comm_dir / f'{zip_path.stem}_binarized_png'
    binarized_dir.mkdir(exist_ok=True)

    # Create the ocr directory
    lines_dir = comm_dir / 'lines'
    lines_dir.mkdir(exist_ok=True)

    # Move the zip file to the new directory
    # zip_path = zip_path.rename(comm_dir / zip_path.name)

    # Unzip the file using python
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(comm_dir)

    # Process the images
    for img_path in tqdm(sorted(img_dir.glob(f'*.{img_extension}'))):
        # Binarize the image and convert it to png

        if (binarized_dir / img_path.with_suffix('.png').name).exists() and (lines_dir / f'{img_path.stem}.json').exists():
            continue
        try:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            img_path = binarized_dir / img_path.with_suffix('.png').name
            cv2.imwrite(str(img_path), img)
            del img

            # Get the lines using kraken's legacy model
            lines = line_detection_model.predict(img_path)

            # Write the results to a file
            output = [line.xyxy for line in lines]
        except Exception as e:
            logger = get_ajmc_logger('segment_ia_commentaries')
            logger.error(f'Error processing {img_path}: {e}')
            output = []

        with open(lines_dir / f'{img_path.stem}.json', 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    # Remove the unzipped files
    shutil.rmtree(img_dir)
