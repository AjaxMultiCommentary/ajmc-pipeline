import re
from collections import Counter

from ajmc.ajmc_corpora import variables as vs

raw_path = vs.BASE_STORING_DIR / 'the_latin_library/data/corpus_cleaned.txt'
cleaned_path = raw_path.parent / 'corpus_cleaned.txt'

text = raw_path.read_text(encoding='utf-8')

text = re.sub(r'\n[0-9XVILC ]+\n', '\n', text)
text = re.sub(r'Neo-Latin', '', text)
text = re.sub(r'Christian Latin', '', text)
text = re.sub(r'The Latin Library', '', text)
text = re.sub(r'The Classics Page', '', text)
text = re.sub(r'Medieval Latin', '', text)

cleaned_path.write_text(text, encoding='utf-8')

len(re.findall('The Latin Library', text))
#%%
raw_lines = [line for line in text.split('\n')]
line_counts = Counter([line for line in raw_lines if line.strip()])
#%%
threshold = 5
discard_counts = {line: count for line, count in line_counts.items() if count > threshold}
sorted_discard_counts = sorted(discard_counts.items(), key=lambda x: x[1], reverse=True)

for line, count in sorted_discard_counts:
    print(count, ' - ', line)

print(line_counts.most_common(100))
#%%
to_discard = [line for line, count in line_counts.items() if count > threshold]

#%%
cleaned_lines = [l for l in raw_lines if l not in to_discard]
cleaned_lines = [l for l in cleaned_lines if l.strip() or l == '']
text = '\n'.join(cleaned_lines)
text = re.sub(r'\n{3,}', '||||', text)
text = re.sub(r'\n\n', '@@@@', text)
text = re.sub(r'\n', ' ', text)
text = re.sub(r'\s+', ' ', text)
text = re.sub(r'@@@@', '\n\n', text)
text = re.sub(r'\|\|\|\|', '\n\n\n', text)

cleaned_path.write_text(text, encoding='utf-8')
