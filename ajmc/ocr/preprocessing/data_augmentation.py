import argparse
import random
from pathlib import Path

from PIL import ImageFilter, Image
from tqdm import tqdm

from ajmc.ocr.pytorch import data_generation as dg


def random_distort(img: 'PIL.Image') -> 'PIL.Image':
    distort = random.gauss(2.8, 0.5)
    distort = max(min(distort, 4), 1.5)
    sigma = random.choices([9, 10, 11], weights=[0.1, 0.8, 0.1])[0]
    eps = random.choices([random.uniform(0.01, 0.1), 0.03], weights=[0.2, 0.8])[0]
    try:
        distorted = dg.distort_line(img, distort=distort, sigma=sigma, eps=eps)
    except:
        return img

    if 0.85 * img.size[1] < distorted.size[1] < 1.7 * img.size[1]:  # To avoid weird kraken truncation and whiteboards
        return distorted
    else:
        return img


def random_degrade(img: 'PIL.Image') -> 'PIL.Image':
    eta = random.choices([random.uniform(0, 0.05), 0], weights=[0.07, 0.93])[0]
    alpha = random.gauss(1, 0.5)
    alpha = max(min(alpha, 3), 0.1)
    beta = random.choices([random.uniform(0.01, 0.15), 1.5], weights=[0.3, 0.7])[0]

    try:
        return dg.degrade_line(img, eta=eta, alpha=alpha, beta=beta)
    except:
        return img


def erode(image: 'PIL.Image', size: int = 3) -> 'PIL.Image':
    return image.filter(ImageFilter.MaxFilter(size))


def write_augmented_line(img_path: Path,
                         output_dir: Path):
    image = Image.open(img_path).convert('L')
    chance = random.randint(0, 100)

    if chance < 10:
        random_distort(image).save((output_dir / (img_path.stem + '_dis.png')))
    elif chance < 35:
        random_degrade(random_distort(image)).save((output_dir / (img_path.stem + '_dis_deg.png')))
    elif chance < 45:
        erode(random_distort(image), size=3).save((output_dir / (img_path.stem + '_dis_ero.png')))
    elif chance < 60:
        erode(random_degrade(random_distort(image)), size=3).save((output_dir / (img_path.stem + '_dis_deg_ero.png')))
    elif chance < 75:
        random_degrade(image).save((output_dir / (img_path.stem + '_deg.png')))
    elif chance < 90:
        erode(random_degrade(image), size=3).save((output_dir / (img_path.stem + '_deg_ero.png')))
    else:
        erode(image, size=3).save((output_dir / (img_path.stem + '_ero.png')))


def main(img_dir: Path, output_dir: Path):
    for image_path in tqdm(img_dir.glob('*.png')):
        txt_path = image_path.with_suffix('.txt')
        (output_dir / txt_path.name).write_text(txt_path.read_text('utf-8'), encoding='utf-8')
        write_augmented_line(image_path, output_dir)  # coucou
        write_augmented_line(image_path, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=Path)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True, parents=True)
    main(args.img_dir, args.output_dir)
