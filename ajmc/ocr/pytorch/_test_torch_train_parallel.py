from ajmc.ocr.pytorch import train_parallel as tp
from tests.test_ocr import sample_objects as so

# from tests.test_ocr import sample_objects as so


if __name__ == '__main__':
    config = so.get_sample_config(mode='gpu')
    so.get_and_write_sample_dataset(50, config)

    tp.main(config=config)
