"""Use this script to replace a text in all via-files."""
import os
from ajmc.commons.miscellaneous import walk_dirs
from ajmc.commons.variables import PATHS

old_pattern = ''
new_pattern = ''

for dir_ in walk_dirs(PATHS['base_dir']):
    via_path = os.path.join(PATHS['base_dir'], dir_, PATHS['via_path'])
    if os.path.exists(via_path):
        with open(via_path, 'r') as f:
            text = f.read()

        text = text.replace(old_pattern, new_pattern)

        with open(via_path, 'w') as f:
            f.write(text)
