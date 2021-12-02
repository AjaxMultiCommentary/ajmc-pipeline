⚠️ This is not up-to-date ⚠️

# A few words on the code

`evaluator` performs regional and coordinates-based CER and WER evaluation between Lace groundtruth-files and Lace or OCR-D ocr-files.

**Regional** means that both CER and WER can be assessed region per region (e.g. primary text, commentary, footnotes....).

**Coordinates-based** means that evaluation does not process documents in a linear manner, which is prone to alignement error when document layouts are complex. Instead, `evaluator` matches overlapping segments (e.g. words) in groundtruth and ocr-data.

More formally, for each segment $s$ in groundtruth data, `evaluator`:
- finds the segments $s'$ in ocr-data that overlaps the most with $s$. 
- calculate Levenshtein distance between $s$ and $s'$

# Output

`evaluator` outputs : 

- a .tsv file containing regional and global CER and WER scores
- a .tsv file containing all edit operations, sorted by decreasing frequency
- .html files recreating each analysed page and allowing the user to visually compare OCR with groundtruth. 
- if argument `draw_rectangles` is passed, copies of the images with each segments outbounding rectangle.




