"""A bunch of ocr tools."""

from pathlib import Path
from typing import Union

from ajmc.commons import variables as vs
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.file_management import get_62_based_datecode


@docstring_formatter(**docstrings)
def get_kraken_command(commentary_id: str, model_path: Union[str, Path]) -> str:
    """LEGACY. Returns the command to be executed by Kraken.

    Args:
        commentary_id: {commentary_id}
        model_path: the path to the model to be used.
    """

    ocr_outputs_dir = vs.get_comm_ocr_runs_dir(commentary_id) / (get_62_based_datecode() + '_' + model_path.stem)
    ocr_outputs_dir.mkdir(parents=True, exist_ok=True)

    img_dir = vs.get_comm_img_dir(commentary_id)
    img_paths = sorted([p for p in img_dir.glob(f'*{vs.DEFAULT_IMG_EXTENSION}')])
    ocr_paths = [(ocr_outputs_dir / p.name).with_suffix('.hocr') for p in img_paths]

    file_list = ' '.join([f'-i {img} {ocr}' for img, ocr in zip(img_paths, ocr_paths)])

    return f'kraken {file_list} -h segment ocr --model model_path'


@docstring_formatter(**docstrings)
def guess_ocr_format(ocr_path: Path) -> str:
    """Guesses the ocr-format of a file.

    Args:
        ocr_path: {ocr_path}

    Returns:
        The ocr-format of the file, either 'pagexml', 'krakenhocr' or 'tesshocr'.
    """

    if ocr_path.suffix.endswith('xml'):
        return 'pagexml'
    elif ocr_path.suffix == '.html':
        return 'krakenhocr'
    elif ocr_path.suffix == '.hocr':
        return 'tesshocr'
    else:
        raise NotImplementedError("""The format could not be identified. It looks like the format is neither 
        `tesshocr`, nor `krakenhocr` nor `pagexml`, which are the only formats this module deals with.""")
