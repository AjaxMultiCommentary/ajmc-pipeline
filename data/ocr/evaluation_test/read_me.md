This directory contains a groundtruth (gt) page and a modified (test) page, in which errors have 
been intentionally added. The errors are the following: 

| charset   | replacement | deletion | insertion | total |
| --------- | ----------- | -------- | --------- | ----- |
| latin     | 2           | 4        | 3         | 9     |
| greek     | 12          | 5        | 4         | 21    |
| numbers   | 4           | 1        | 2         | 7     |
| **total** | 18          | 10       | 9         | 37    |

These errors spread over `25` words.

The page contains a total of :

```python
from ajmc.text_processing.ocr_classes import OcrPage

page = OcrPage('sophoclesplaysa05campgoog_0146',
               ocr_path='/data/ocr/evaluation_test/gt_sophoclesplaysa05campgoog_0146.html')
print('Wordcount:  ', len(page.children.words))
print('Charcount:  ', sum([len(w.text) for w in page.children.words]))
```

returns:
```
>>> Wordcount:   548
>>> Charcount:  2518
```

Which implies that evaluating this page should yield the following results (see tests):

- **CER** should be `37/2518 = 0.014694201747418586` (**NLD** `0.9853057982525815`)
- **WER** should be `25/548 = 0.04562043795620438`

