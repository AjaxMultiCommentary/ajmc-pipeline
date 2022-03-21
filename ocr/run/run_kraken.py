import os

commentary = 'lestragdiesdeso00tourgoog'
base_path = ''
png_path = ''
ocr_path = ''


def prepare_filelists():
    png_abs_path = os.path.join(base_path, commentary, png_path)
    image_names = [fname for fname in os.listdir(png_abs_path) if fname.endswith('.png')]
    image_paths = [os.path.join(png_abs_path, f) for f in image_names]
    ocr_paths = [os.path.join(ocr_path, f[:-3] + 'hocr') for f in image_paths]
