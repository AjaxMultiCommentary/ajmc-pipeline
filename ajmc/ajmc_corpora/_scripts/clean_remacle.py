import re
from collections import Counter
from pathlib import Path

raw_path = Path('//mnt/ajmcdata1/data/remacle/data/corpus.txt')
cleaned_path = raw_path.parent / 'corpus_cleaned.txt'

text = raw_path.read_text(encoding='utf-8')
text = re.sub(r'%PDF.*?%%EOF', '', text, flags=re.DOTALL)

raw_lines = [line for line in text.split('\n')]
line_counts = Counter([line for line in raw_lines if line.strip()])
#%%
threshold = 60
discard_counts = {line: count for line, count in line_counts.items() if count > threshold}
sorted_discard_counts = sorted(discard_counts.items(), key=lambda x: x[1], reverse=True)

for line, count in sorted_discard_counts:
    print(count, ' - ', line)

#%%
to_discard = [line for line, count in line_counts.items() if (count > threshold and len(line.split()) > 1) or (len(line.split()) > 15 and count > 2)]

print(to_discard)

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
# remove http links
text = re.sub(r'http\S+', '', text)

cleaned_path.write_text(text, encoding='utf-8')
