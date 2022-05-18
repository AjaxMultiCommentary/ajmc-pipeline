from ajmc.commons import variables
from ajmc.text_importation.classes import Commentary, Page
import os
import re


def rename_file(path: str, pattern: str, replacement: str):
    command = "rename -f 's/{}/{}/' '{}'".format(pattern, replacement, path)
    os.system(command)


def mv_file(old: str, new: str):
    command = "mv '{}' '{}'".format(old, new)
    print(command)
    os.system(command)


commentary_id = 'Kamerbeek1953'
ocr_dir = '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/Kamerbeek1953/ocr/runs/17u09o_kraken/outputs'
new_dir = '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/Kamerbeek1953/ocr/runs/17u09o_kraken/outputs_'
# %% Rename images

img_dir = os.path.join(variables.PATHS['base_dir'], commentary_id, variables.PATHS['png'])
# img_dir = '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/Kamerbeek1953/images/png_binary'
fnames = [fname for fname in os.listdir(img_dir) if fname.endswith('.png')]

for fname in fnames:
    # Reformulate filenames
    id_, end = fname.split('_')
    new_name = id_ + '_' + end[1:]

    # Create the path
    old_path = os.path.join(img_dir, fname)
    new_path = os.path.join(img_dir, new_name)

    # moves files
    mv_file(old_path, new_path)

# %% rename files in via_dict

# Import the via_dict
via_path = os.path.join(variables.PATHS['base_dir'], commentary_id, variables.PATHS['via_path'])
with open(via_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Rename the old via
archived_via_path = via_path[:-5] + '_old.json'
mv_file(via_path, archived_via_path)

# Do the replacement and write a new file
pattern = re.compile(commentary_id + '_0([0-9]{4})\.')
text_mod = re.sub(pattern=pattern, repl=commentary_id + r'_\1.', string=text)
with open(via_path, 'w', encoding='utf-8') as f:
    f.write(text_mod)

# %% test it
comm = Commentary.from_folder_structure(ocr_dir=ocr_dir)
comm_ = Commentary(
    via_path=archived_via_path,
    ocr_dir=ocr_dir)

assert [p.id for p in comm.olr_groundtruth_pages] == [p.id.split('_')[0] + '_' + p.id.split('_')[1][1:] for p in
                                                      comm_.olr_groundtruth_pages]

# %% Rename ocr outputs
fnames = [fname for fname in os.listdir(ocr_dir) if fname.endswith('.hocr')]

for fname in fnames:
    # Reformulate filenames
    id_, end = fname.split('_')
    new_name = id_ + '_' + end[1:]

    old_path = os.path.join(ocr_dir, fname)
    new_path = os.path.join(new_dir, new_name)

    with open(old_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = text.replace(fname.split('.')[0], new_name.split('.')[0])

    with open(new_path, 'w', encoding='utf-8') as f:
        f.write(text)

# %% Testing if modified HOCR files still work

orig_page = Page(
    ocr_path='/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/Kamerbeek1953/ocr/runs/17u09o_kraken/outputs_/Kamerbeek1953_0101.hocr')
new_page = Page(
    ocr_path='/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/Kamerbeek1953/ocr/runs/17u09o_kraken/outputs/Kamerbeek1953_00101.hocr')

new_page.text == orig_page.text
