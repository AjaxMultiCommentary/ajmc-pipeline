import argparse
import random
from pathlib import Path

import numpy as np
from PIL import ImageFilter, Image, ImageOps
from scipy.ndimage import distance_transform_cdt, binary_closing, affine_transform, gaussian_filter, geometric_transform
from tqdm import tqdm

from ajmc.ocr.data_processing.data_generation import logger


def pil2array(img: Image.Image) -> np.ndarray:
    """Kraken linegen code."""
    if img.mode == '1':
        return np.array(img.convert('L'))
    return np.array(img)


def array2pil(a: np.ndarray) -> Image.Image:
    if a.dtype == np.dtype("B"):
        if a.ndim == 2:
            return Image.frombytes("L", (a.shape[1], a.shape[0]),
                                   a.tobytes())
        elif a.ndim == 3:
            return Image.frombytes("RGB", (a.shape[1], a.shape[0]),
                                   a.tobytes())
        else:
            raise Exception("bad image rank")
    elif a.dtype == np.dtype('float32'):
        return Image.frombytes("F", (a.shape[1], a.shape[0]), a.tobytes())
    else:
        raise Exception("unknown image type")


def degrade_line(img: Image.Image, eta=0.0, alpha=1.5, beta=1.5, alpha_0=1.0, beta_0=1.0):
    """
    Degrades a line image by adding noise.

    For parameter meanings consult "A statistical, nonparametric methodology for document degradation model validation" (2000) by Tanugo et al.

    Args:
        img (PIL.Image): Input image
        eta (float): Between 0 and 1. And jitters to image. Recommend staying to max 0.05
        alpha (float): Seems to be the concentration of the noise around letters. The higher the more concentrated.Go between 0.1 and 3.
        beta (float): the amount of holes in letters. Try a few 0.1, 0.3 and default.
        alpha_0 (float): No clue what these do, leave default or see paper.
        beta_0 (float): No clue what these do, leave default or see paper.

    Returns:
        PIL.Image in mode '1'
    """
    logger.debug('Inverting and normalizing input image')
    img = pil2array(img)
    img = np.amax(img) - img
    img = img * 1.0 / np.amax(img)

    logger.debug('Calculating foreground distance transform')
    fg_dist = distance_transform_cdt(1 - img, metric='taxicab')
    logger.debug('Calculating flip to white probability')
    fg_prob = alpha_0 * np.exp(-alpha * (fg_dist ** 2)) + eta
    fg_prob[img == 1] = 0
    fg_flip = np.random.binomial(1, fg_prob)

    logger.debug('Calculating background distance transform')
    bg_dist = distance_transform_cdt(img, metric='taxicab')
    logger.debug('Calculating flip to black probability')
    bg_prob = beta_0 * np.exp(-beta * (bg_dist ** 2)) + eta
    bg_prob[img == 0] = 0
    bg_flip = np.random.binomial(1, bg_prob)

    # flip
    logger.debug('Flipping')
    img -= bg_flip
    img += fg_flip

    logger.debug('Binary closing')
    sel = np.array([[1, 1], [1, 1]])
    img = binary_closing(img, sel)
    logger.debug('Converting to image')
    return array2pil(255 - img.astype('B') * 255)


def distort_line(img: Image.Image, distort=3.0, sigma=10, eps=0.03, delta=0.3):
    """
    Distorts a line image.

    Run BEFORE degrade_line as a white border of 5 pixels will be added.

    Args:
        img (PIL.Image): Input image
        distort (float): Distorting of the image set between 1.5 and 4.0, with majority around 2.5
        sigma (float): distorting of the strokes, set between 0.5, 1.5 for 5% of image, else default
        eps (float): set default to 80%, else random between 0.01 and 0.1
        delta (float):

    Returns:
        PIL.Image in mode 'L'
    """
    w, h = img.size
    # XXX: determine correct output shape from transformation matrices instead
    # of guesstimating.
    logger.debug('Pasting source image into canvas')
    image = Image.new('L', (int(1.5 * w), 4 * h), 255)
    image.paste(img, (int((image.size[0] - w) / 2), int((image.size[1] - h) / 2)))
    line = pil2array(image.convert('L'))

    # shear in y direction with factor eps * randn(), scaling with 1 + eps *
    # randn() in x/y axis (all offset at d)
    logger.debug('Performing affine transformation')
    m = np.array([[1 + eps * np.random.randn(), 0.0], [eps * np.random.randn(), 1.0 + eps * np.random.randn()]])
    c = np.array([w / 2.0, h / 2])
    d = c - np.dot(m, c) + np.array([np.random.randn() * delta, np.random.randn() * delta])
    line = affine_transform(line, m, offset=d, order=1, mode='constant', cval=255)

    hs = gaussian_filter(np.random.randn(4 * h, int(1.5 * w)), sigma)
    ws = gaussian_filter(np.random.randn(4 * h, int(1.5 * w)), sigma)
    hs *= distort / np.amax(hs)
    ws *= distort / np.amax(ws)

    def _f(p):
        return p[0] + hs[p[0], p[1]], p[1] + ws[p[0], p[1]]

    logger.debug('Performing geometric transformation')
    img = array2pil(geometric_transform(line, _f, order=1, mode='nearest'))
    logger.debug('Cropping canvas to content box')
    img = img.crop(ImageOps.invert(img).getbbox())
    return img


def random_distort(img: Image.Image) -> Image.Image:
    """Distort the image with a random distortion factor, sigma and epsilon."""
    distort = random.gauss(2.8, 0.5)
    distort = max(min(distort, 4), 1.5)
    sigma = random.choices([9, 10, 11], weights=[0.1, 0.8, 0.1])[0]
    eps = random.choices([random.uniform(0.01, 0.1), 0.03], weights=[0.2, 0.8])[0]
    try:
        distorted = distort_line(img, distort=distort, sigma=sigma, eps=eps)
    except:
        return img

    if 0.85 * img.size[1] < distorted.size[1] < 1.7 * img.size[1]:  # To avoid weird kraken truncation and whiteboards
        return distorted
    else:
        return img


def random_degrade(img: Image.Image) -> Image.Image:
    """Degrade the image with a random eta, alpha and beta."""
    eta = random.choices([random.uniform(0, 0.05), 0], weights=[0.07, 0.93])[0]
    alpha = random.gauss(1, 0.5)
    alpha = max(min(alpha, 3), 0.1)
    beta = random.choices([random.uniform(0.01, 0.15), 1.5], weights=[0.3, 0.7])[0]

    try:
        return degrade_line(img, eta=eta, alpha=alpha, beta=beta)
    except:
        return img


def erode(image: Image.Image, size: int = 3) -> Image.Image:
    return image.filter(ImageFilter.MaxFilter(size))


def random_augment_line(img_path: Path,
                        output_dir: Path):
    """Augment a line image with a random augmentation method."""

    image = Image.open(img_path).convert('L')
    chance = random.randint(0, 100)

    if chance < 10:
        random_distort(image).save((output_dir / (img_path.stem + '_dis.png')))
        (output_dir / (img_path.stem + '_dis.txt')).write_text(img_path.with_suffix('.txt').read_text('utf-8'), encoding='utf-8')

    elif chance < 35:
        random_degrade(random_distort(image)).save((output_dir / (img_path.stem + '_dis_deg.png')))
        (output_dir / (img_path.stem + '_dis_deg.txt')).write_text(img_path.with_suffix('.txt').read_text('utf-8'), encoding='utf-8')

    elif chance < 45:
        erode(random_distort(image), size=3).save((output_dir / (img_path.stem + '_dis_ero.png')))
        (output_dir / (img_path.stem + '_dis_ero.txt')).write_text(img_path.with_suffix('.txt').read_text('utf-8'), encoding='utf-8')

    elif chance < 60:
        erode(random_degrade(random_distort(image)), size=3).save((output_dir / (img_path.stem + '_dis_deg_ero.png')))
        (output_dir / (img_path.stem + '_dis_deg_ero.txt')).write_text(img_path.with_suffix('.txt').read_text('utf-8'), encoding='utf-8')

    elif chance < 75:
        random_degrade(image).save((output_dir / (img_path.stem + '_deg.png')))
        (output_dir / (img_path.stem + '_deg.txt')).write_text(img_path.with_suffix('.txt').read_text('utf-8'), encoding='utf-8')

    elif chance < 90:
        erode(random_degrade(image), size=3).save((output_dir / (img_path.stem + '_deg_ero.png')))
        (output_dir / (img_path.stem + '_deg_ero.txt')).write_text(img_path.with_suffix('.txt').read_text('utf-8'), encoding='utf-8')

    else:
        erode(image, size=3).save((output_dir / (img_path.stem + '_ero.png')))
        (output_dir / (img_path.stem + '_ero.txt')).write_text(img_path.with_suffix('.txt').read_text('utf-8'), encoding='utf-8')


def augment_img_dir(img_dir: Path, output_dir: Path):
    for image_path in tqdm(img_dir.glob('*en.png')):
        random_augment_line(image_path, output_dir)
        random_augment_line(image_path, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=Path)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    augment_img_dir(args.img_dir, args.output_dir)
