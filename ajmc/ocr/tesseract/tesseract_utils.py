import os
import pathlib
import shutil
from pathlib import Path
import os

import Levenshtein
from typing import Tuple

import pandas as pd

from ajmc.commons.arithmetic import safe_divide
from ajmc.ocr.evaluation.utils import record_editops, write_editops_record
from ajmc.ocr.utils import harmonise_unicode
from ajmc.commons.file_management.utils import get_62_based_datecode
from ajmc.ocr.preprocessing.data_preparation import resize_ocr_dataset


def run_tesseract(img_dir: str,
                  output_dir: str,
                  langs: str,
                  config: dict = None,
                  psm: int = 3,
                  img_suffix: str = '.png',
                  tessdata_prefix: str = '/Users/sven/packages/tesseract/tessdata/'
                  ):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the config
    if config:
        (output_dir / 'tess_config').write_text('\n'.join([f'{k} {v}' for k, v in config.items()]))

    command = f"""cd {img_dir}; export TESSDATA_PREFIX={tessdata_prefix}; \
for i in *{img_suffix} ; \
do tesseract "$i" "{output_dir}/${{i::${{#i}}-4}}" \
-l {langs} \
--psm {psm} \
{os.path.join(output_dir, 'tess_config') if config else ''}; \
done;"""

    # Writes the command to remember how this was run
    with open(os.path.join(output_dir, 'command.sh'), 'w') as f:
        f.write(command)

    # Write the data related metadata
    if os.path.exists(os.path.join(img_dir, 'metadata.json')):
        shutil.copyfile(os.path.join(img_dir, 'metadata.json'), os.path.join(output_dir, 'data_metadata.json'))

    # Run the command
    os.system(command=command)


def evaluate_tesseract(gt_dir: str,
                       ocr_dir: str,
                       gt_suffix: str = '.gt.txt',
                       ocr_suffix: str = '.txt',
                       error_record: dict = None,
                       editops_record: dict = None,
                       write_to_file: bool = True, ) -> Tuple[dict, dict]:
    error_record = error_record if error_record else {k: [] for k in ['id', 'gt', 'ocr',
                                                                      'chars', 'chars_distance',
                                                                      'words', 'words_distance']}
    editops_record = editops_record if editops_record else {}
    ocr_dir, gt_dir = Path(ocr_dir), Path(gt_dir)

    for ocr_path in ocr_dir.glob(f'*{ocr_suffix}'):
        gt_path = gt_dir / ocr_path.with_suffix(gt_suffix).name
        gt_text = gt_path.read_text(encoding='utf-8')
        ocr_text = ocr_path.read_text(encoding='utf-8')

        # Postprocess the OCR text
        ocr_text = ocr_text.strip('\n')
        ocr_text = ocr_text.strip()
        ocr_text = harmonise_unicode(ocr_text)
        gt_text = gt_text.strip(' ')
        gt_text = harmonise_unicode(gt_text)

        # compute distance
        error_record['id'].append(ocr_path.stem)
        error_record['gt'].append(gt_text)
        error_record['ocr'].append(ocr_text)
        error_record['chars'].append(len(gt_text))
        error_record['chars_distance'].append(Levenshtein.distance(gt_text, ocr_text))
        error_record['words'].append(len(gt_text.split(' ')))
        error_record['words_distance'].append(Levenshtein.distance(gt_text.split(), ocr_text.split()))

        # Record edit operations
        editops_record = record_editops(gt_word=gt_text,
                                        ocr_word=ocr_text,
                                        editops=Levenshtein.editops(ocr_text, gt_text),
                                        editops_record=editops_record)

    cer = round(safe_divide(sum(error_record['chars_distance']), sum(error_record['chars'])), 3)
    wer = round(safe_divide(sum(error_record['words_distance']), sum(error_record['words'])), 3)

    print(f'Character Error Rate: {cer}')
    print(f'Word Error Rate: {wer}')

    # Write files
    if write_to_file:
        eval_dir = ocr_dir.parent / 'evaluation'
        os.makedirs(eval_dir, exist_ok=True)

        editops_record = {k: v for k, v in sorted(editops_record.items(), key=lambda item: item[1], reverse=True)}
        write_editops_record(editops_record=editops_record, output_dir=eval_dir)

        pd.DataFrame.from_dict(error_record, orient='columns').to_csv(os.path.join(eval_dir, 'error_record.tsv'),
                                                                      sep='\t',
                                                                      index=False)

        with open(os.path.join(eval_dir, 'results.txt'), 'w') as f:
            f.write(f'cer\twer\n{cer}\t{wer}')

    return error_record, editops_record


def reformulate_output_dir(output_dir: str) -> pathlib.Path:
    output_dir = Path(output_dir)
    return output_dir.parent / f'{get_62_based_datecode()}_{output_dir.name}/outputs'


def create_general_table(xps_root):
    xps_root = Path(xps_root)
    general_table = pd.DataFrame()
    for xp_dir in xps_root.glob('*'):
        if xp_dir.is_dir():
            xp_name = xp_dir.name
            resizing = [par for par in xp_name.split('_') if 'rsz' in par]
            resizing = int(resizing[0][3:]) if resizing else None
            model = xp_name.split('_')[1]
            psm = [par for par in xp_name.split('_') if 'psm' in par]
            psm = int(psm[0][3:]) if psm else 7

            if (xp_dir / 'evaluation' / 'results.txt').exists():
                with open(xp_dir / 'evaluation' / 'results.txt', 'r') as f:
                    cer, wer = f.read().split('\n')[1].split('\t')
                general_table = pd.concat([general_table,
                                           pd.DataFrame({'xp_name': [xp_name],
                                                         'models': [model],
                                                         'data': [None],
                                                         'psm': [psm],
                                                         'resize': [resizing],
                                                         'cer': [cer],
                                                         'wer': [wer]})
                                           ])

    general_table.to_csv(xps_root / 'general_table.tsv', sep='\t', index=False)
    return general_table


