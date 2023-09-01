from ajmc.ocr.pytorch import model as m
from tests.test_ocr import sample_objects as so
from tests.test_ocr.test_torch_data_processing import build_dataset

# We need to :
config = so.get_sample_config()
single_img_tensor = so.get_single_img_tensor(config)

dataset = build_dataset(4)
test_batch = next(iter(dataset))

model = m.OcrTorchModel(config)

out = model.forward(test_batch[0], test_batch[2])

from pathlib import Path
from ajmc.commons.variables import get_comm_ocr_gt_pairs_dir
import shutil

comm_id = 'sophoclesplaysa05campgoog'
pairs_dir = get_comm_ocr_gt_pairs_dir(comm_id)
output_dir = Path(f'/Users/sven/Desktop/tests/{comm_id}_test')
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir(exist_ok=True, parents=True)
