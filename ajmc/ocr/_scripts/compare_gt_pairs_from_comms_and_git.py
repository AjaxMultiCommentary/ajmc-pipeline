"""Compare canonical generated ocr gt lines of PD commentaries with GT-commentaries-ocr"""
import os
import unicodedata
from pathlib import Path

import Levenshtein

from ajmc.commons import variables as vs

git_gt_dir = Path('/Users/sven/Desktop/tess_xps/data/GT-commentaries-OCR/data/')

check_dir = Path('/Users/sven/Desktop/checks/')
for git_comm_dir in git_gt_dir.glob('*'):
    if git_comm_dir.is_dir():
        comm_id = git_comm_dir.name
        can_gt_dir = vs.get_comm_ocr_gt_pairs_dir(comm_id)
        git_comm_dir = git_comm_dir / 'GT-pairs'

        print('\n\nProcessing ', comm_id)
        print('The number of lines in git is : ', len(list(git_comm_dir.glob('*.gt.txt'))))
        print('The number of lines in canonical is : ', len(list(can_gt_dir.glob('*.txt'))))

        git_pages = set([p.stem.split('_')[1] for p in git_comm_dir.glob('*.gt.txt')])
        can_pages = set([p.stem.split('_')[1] for p in can_gt_dir.glob('*.txt')])

        print('The number of pages in git is : ', len(git_pages))
        print('The number of pages in canonical is : ', len(can_pages))

        print('git pages : ', git_pages)
        print('can pages : ', can_pages)

        git_lines = [unicodedata.normalize('NFC', p.read_text('utf-8')) for p in sorted(git_comm_dir.glob('*.gt.txt'))]
        can_lines = [(p.name, p.read_text('utf-8')) for p in sorted(can_gt_dir.glob('*.txt'))]

        for id_, line in can_lines:
            if not unicodedata.is_normalized('NFC', line):
                print(id_, line)

        for id_, line in can_lines:
            if line not in git_lines:
                # copy line image to a folder
                command = f"cp {can_gt_dir / id_.replace('.gt.txt', '.png')} {check_dir / id_}.png"
                os.system(command)

                # Write line text to a file
                (check_dir / id_).write_text(line, 'utf-8')

                best_git_line = min(git_lines, key=lambda x: Levenshtein.distance(x, line))
                print('\nline mismatch : ', id_)
                # print('        0         1         2         3         4         5         6         7         8         9')
                print('can : ', f"|{line}|")
                print('git : ', f"|{best_git_line}|")
                # print(Levenshtein.editops(line, best_git_line))
