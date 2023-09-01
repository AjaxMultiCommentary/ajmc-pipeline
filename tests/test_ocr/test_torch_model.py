from ajmc.ocr.pytorch import model as m
from tests.test_ocr import sample_objects as so


config = so.get_sample_config()
test_model = m.OcrTorchModel(config=config)
