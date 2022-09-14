`ocr/evaluation` performs a double evaluation of ocr outputs against given groundtruth data. 

1. **Bag-of-word evaluation**: computes errors by matching words which have the minimal edit distance in a bag of groundtruth and in a a bag of predicted word. 
2. **Coordinate based evaluation**: computes errors by matching words with overlapping coordinates.


# API

To run the evaluation of a commentary, please do :

```python
from ajmc.text_processing import Commentary
from ajmc.ocr.evaluation.evaluation_methods import commentary_evaluation

commentary = Commentary('sophoclesplaysa05campgoog')
bow_error_counts, coord_error_counts, editops = commentary_evaluation(commentary)
```


# Command line interface

⚠️ **CLI usage is not implemented yet** ⚠️

Run `python ocr/evaluation` with the following arguments : 

```shell script
usage: evaluation [-h] [--IMG_DIR IMG_DIR] [--SVG_DIR SVG_DIR]
                 [--OUTPUT_DIR OUTPUT_DIR] [--via_project VIA_PROJECT]
                 [--GROUNDTRUTH_DIR GROUNDTRUTH_DIR] [--OCR_DIR OCR_DIR]
                 [--evaluation_level EVALUATION_LEVEL]
                 [--PARENT_DIR PARENT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --IMG_DIR IMG_DIR     Absolute path to the directory where the image-files
                        are stored
  --SVG_DIR SVG_DIR     Absolute path to the directory where the svg-files are
                        stored. Must be given if `--via_project` is not
  --OUTPUT_DIR OUTPUT_DIR
                        Absolute path to the directory in which outputs are to
                        be stored
  --via_project VIA_PROJECT
                        Absolute path to the .json via project. Must be given
                        if `--SVG_DIR` is not.
  --GROUNDTRUTH_DIR GROUNDTRUTH_DIR
                        Absolute path to the directory in which groundtruth-
                        files are stored
  --OCR_DIR OCR_DIR     Absolute path to the directory in which ocr-files are
                        stored
```

For example : 
```
python evaluator --IMG_DIR "data/test_png" \
--SVG_DIR "data/test_svg" \
--OUTPUT_DIR "output" \
--OCR_DIR "test/"
--GROUNDTRUTH_DIR "test_gt/"
--draw_bboxes
```