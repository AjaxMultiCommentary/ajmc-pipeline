"""Use this script to replace a text in all via-files."""
from ajmc.commons import variables as vs
# CHECKED 2023-01-23
from ajmc.commons.file_management.utils import walk_dirs

old_pattern = ''
new_pattern = ''

for dir_ in walk_dirs(vs.COMMS_DATA_DIR):
    via_path = vs.get_comm_via_path(dir_.name)
    if via_path.exists():
        text = via_path.read_text(encoding='utf-8')
        text = text.replace(old_pattern, new_pattern)
        print(f'Writing {via_path}')
        via_path.write_text(text, encoding='utf-8')

