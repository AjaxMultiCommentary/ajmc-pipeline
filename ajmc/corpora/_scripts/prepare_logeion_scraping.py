import json
from pathlib import Path

divisions_lat = [
    '-met',
    'D',
    'I',
    'N',
    'S',
    'zyzanium'
]

divisions_grc = [
    'Ϝ',
    'Δ',
    'Η',
    'Λ',
    'Ο',
    'Τ',
    '᾽Ρᾶρος'

]

root_dir = Path('/home/najem/logeion')
root_dir.mkdir(exist_ok=True, parents=True)

for division in [divisions_lat, divisions_grc]:
    for i in range(len(division) - 1):
        offset = {'start': division[i], 'stop': division[i + 1]}
        sub_dir = (root_dir / offset['start'])
        sub_dir.mkdir(exist_ok=True, parents=True)
        (sub_dir / 'metadata.json').write_text(json.dumps(offset, ensure_ascii=False), encoding='utf-8')
        (sub_dir / 'words.txt').write_text(offset['start'] + '\n', encoding='utf-8')
