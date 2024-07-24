# This just cleans persee of paragraphs/documents containing greek stopwords.
from pathlib import Path

from tqdm import tqdm

plain_text_path = Path('/mnt/ajmcdata1/data/persee/plaintext.txt')

text = plain_text_path.read_text(encoding='utf-8')

text = text.split('\n\n\n')
latinised_greek_stopwords = ['kai', 'kal', 'tovto', 'tov', 'yap']

to_delete = set()
for i in tqdm(range(len(text))):
    words = text[i].split()
    if any([word.lower() in latinised_greek_stopwords for word in words]):
        to_delete.add(i - 1)
        to_delete.add(i)
        to_delete.add(i + 1)

to_keep = set(range(len(text))) - to_delete

final_text = '\n\n\n'.join([text[i] for i in to_keep])

plain_text_path.write_text(final_text, encoding='utf-8')
print('Deleted', len(to_delete), 'documents.')
