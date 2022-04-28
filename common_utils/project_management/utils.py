from string import ascii_letters
from datetime import datetime

numbers_letters = '0123456789'+ascii_letters


def int_to_62_based_code(number: int) -> str:
    code = ''
    for i in [238328, 3844, 62]:  # 62**3, 62**2, 62
        num = number//i
        assert num <= 62
        code += numbers_letters[num]
        number = number-(num*i)
    return code


def get_62_based_datecode(date: datetime = datetime.now()):

    datecode = ''

    datecode += str(date.year)[-1]
    datecode += numbers_letters[date.month]
    datecode += numbers_letters[date.day]

    day_seconds = (date - date.replace(hour=0, minute=0, second=0, microsecond=0)).seconds

    datecode += int_to_62_based_code(day_seconds)

    return datecode

