from string import ascii_letters
from datetime import datetime
from commons import variables

NUMBERS_LETTERS = '0123456789' + ascii_letters


def int_to_62_based_code(number: int) -> str:
    code = ''
    for i in [238328, 3844, 62]:  # 62**3, 62**2, 62
        num = number // i
        assert num <= 62
        code += NUMBERS_LETTERS[num]
        number = number - (num * i)
    return code


def get_62_based_datecode(date: datetime = datetime.now()):
    datecode = ''

    datecode += str(date.year)[-1]
    datecode += NUMBERS_LETTERS[date.month]
    datecode += NUMBERS_LETTERS[date.day]

    day_seconds = (date - date.replace(hour=0, minute=0, second=0, microsecond=0)).seconds

    datecode += int_to_62_based_code(day_seconds)

    return datecode


def verify_path_integrity(path: str, path_pattern: str) -> None:
    """Verify the integrity of an `ocr_path` with respect to ajmc's folder structure.
    Args:
        path: The path to be tested
        path_pattern: The pattern to be respected (see `commons.variables.FOLDER_STRUCTURE_PATHS`).
    """
    dirs = path.strip('/').split('/')  # Stripping trailing and leading '/' before splitting
    dirs_pattern = path_pattern.strip('/').split('/')

    for dir_, pattern in zip(reversed(dirs), reversed(dirs_pattern)):
        # Make sure the detected commentary id is known
        if pattern == '[commentary_id]':
            assert dir_ in variables.COMMENTARY_IDS, f"""The commentary id ({dir_}) detected 
            in the provided path ({path}) does not match any known commentary_id. """

        # Skip other placeholder patterns (e.g. '[ocr_run_name]').
        elif pattern[0] == '[' and pattern[-1] == ']':
            continue

        # Verify fixed patterns
        else:
            assert pattern == dir_, f"""The provided path ({path}) does not seem to be compliant with AJMC's 
            folder structure. Please make sure you are observing the following pattern: \n {path_pattern} """
