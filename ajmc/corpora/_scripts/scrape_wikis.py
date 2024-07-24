from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

languages = ['el', 'en', 'fr', 'de', 'it', 'la']

for lang in languages:
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}")

    output_dir = Path(f'/mnt/ajmcdata1/data/wiki_{lang}/data')
    output_dir.mkdir(parents=True, exist_ok=True)

    text = ''
    length = 0
    for article in tqdm(ds['train'], desc=f'Processing {lang}'):
        for line in article['text'].split('\n'):
            if len(line) > 100 and not line.startswith('='):
                text += line + '\n'
                length += len(line)
        if lang != 'la' and length > 100_000_000:
            break

    (output_dir / 'corpus_cleaned.txt').write_text(text, encoding='utf-8')
