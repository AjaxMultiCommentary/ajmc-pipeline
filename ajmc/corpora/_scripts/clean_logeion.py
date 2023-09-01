import json
from pathlib import Path

from ajmc.corpora.cleaning_utils import basic_clean

logeion_dir = Path('/home/najem/logeion')

splits = {'latin': ['met', 'D', 'I', 'N', 'S', ],
          'greek': ['Ϝ', 'Δ', 'Η', 'Λ', 'Ο', 'Τ', ]}

for split_name, split_letters in splits.items():
    split_dir = logeion_dir / split_name
    split_dir.mkdir(exist_ok=True, parents=True)

    words = []
    dicts = {}
    for letter in split_letters:
        letter_dir = logeion_dir / letter
        words.extend([line.strip() for line in (letter_dir / 'words.txt').read_text(encoding='utf-8').split('\n') if line.strip() != ''])

        for dict_path in letter_dir.glob('*.jsonl'):
            if dict_path.stem not in dicts:
                dicts[dict_path.stem] = {}
            for line in dict_path.read_text(encoding='utf-8').split('\n'):
                if line.strip() == '':
                    continue
                entry = json.loads(line)
                dicts[dict_path.stem].update(entry)

            dicts[dict_path.stem] = {key: basic_clean(value) for key, value in dicts[dict_path.stem].items()}

    for dict_ in dicts:
        (split_dir / f'{dict_}.json').write_text(json.dumps(dicts[dict_], ensure_ascii=False), encoding='utf-8')

    (split_dir / 'wordlist.txt').write_text('\n'.join(words), encoding='utf-8')
