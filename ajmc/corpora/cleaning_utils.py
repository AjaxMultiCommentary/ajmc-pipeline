import re
import typing
from collections import Counter
from pathlib import Path


def find_recurrent_lines(path: str,
                         n_first_elements: typing.Optional[int] = None,
                         recurrence_threshold: typing.Optional[int] = None):
    text = Path(path).read_text(encoding='utf-8')
    lines = [line for line in text.split('\n') if line.strip()]
    line_counts = Counter(lines)
    if n_first_elements:
        print(line_counts.most_common(n_first_elements))
    if recurrence_threshold:
        print({line: count for line, count in line_counts.items() if count >= recurrence_threshold})


def basic_clean(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)  # Removes multiple spaces
    text = text.strip()  # Removes leading and trailing whitespace
    text = re.sub(r'https?://\S+', '', text)  # Removes URLs
    return text


def harmonise_linebreaks(text: str) -> str:
    text = re.sub(r'\n{3,}', '||||', text)
    text = re.sub(r'\n\n', '@@@@', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'@@@@', '\n\n', text)
    text = re.sub(r'\|\|\|\|', '\n\n\n', text)
    return text
