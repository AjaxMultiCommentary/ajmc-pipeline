from ajmc.ocr.pytorch import train_parallel as tp
from tests.test_ocr import sample_objects as so

# from tests.test_ocr import sample_objects as so


if __name__ == '__main__':
    config = so.get_sample_config(mode='multi_gpu')
    config['total_steps'] = 1000
    so.get_and_write_sample_dataset(10, config)

    tp.main(config=config)
