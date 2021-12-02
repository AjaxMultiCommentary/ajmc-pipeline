⚠️ This is not up-to-date ⚠️

# Presentation

`oclr` is python-package containing **Optical Character and Layout Recognition utilities**. 

It contains the modules `annotation_helper` and `evaluator`. 
 
`Annotation_helper` performs the following tasks :

1. Converting [Lace's](http://oglediting.chs.harvard.edu/) region annotations (.svg) to [VIA2](https://www.robots.ox.ac.uk/~vgg/software/via/) annotations (.csv)
2. Detecting zones in image-files
3. Adding detected zones to Lace-annotations. 

`evaluator` performs regional and coordinates-based CER and WER evaluation between Lace groundtruth-files and Lace (hOCR) or OCR-D (PAGE-XML) outputs.

This file is general presentation. More detailed information on the code can be found in each subpackage. 

# Setup

Please install `olcr` from source, using `git clone https://github.com/AjaxMultiCommentary/oclr`. 

# Run

In order to run `oclr` properly, you first need to create an environment. `requirements.txt` 
specifies the pip-requirements for creating an environment (for instance with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)). 

Once the virtual environment is created, go to the annotation_helper directory : `cd oclr`.

Activate your environment : `conda activate myenv`. 

Then follow the specific instructions of each program. 


# Run annotation_helper

Run `python3 annotation_helper` with the following arguments : 

```shell script
usage: via_helper [-h] [--IMG_DIR IMG_DIR] [--SVG_DIR SVG_DIR]
                         [--OUTPUT_DIR OUTPUT_DIR] [--via_project VIA_PROJECT]
                         [--dilation_kernel_size DILATION_KERNEL_SIZE]
                         [--dilation_iterations DILATION_ITERATIONS]
                         [--artifact_threshold ARTIFACTS_SIZE_THRESHOLD]
                         [--draw_rectangles] [--merge_zones]

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
  --dilation_kernel_size DILATION_KERNEL_SIZE
                        Dilation kernel size, preferably an odd number. Tweak
                        this parameter and `--dilation_iterations` to improve
                        automatic boxing.
  --dilation_iterations DILATION_ITERATIONS
                        Number of iterations in dilation, default 1
  --artifact_threshold ARTIFACTS_SIZE_THRESHOLD
                        Size-threshold under which contours are to be
                        considered as artifacts and removed, expressed as a
                        percentage of image height. Default is 0.01
  --draw_rectangles     Whether to output images with both shrinked and
                        dilated rectangles. This is usefull if you want to
                        have a look at images, e.g. to test dilation
                        parameters.
  --merge_zones         Whether to add automatically detected zones to Lace-
                        zones before exporting annotation file
```

For example : 
```shell script
python3 via_helper --IMG_DIR "data/test_png" \
--SVG_DIR "data/test_svg" \
--OUTPUT_DIR "output" \
--dilation_kernel_size 51 \
--dilation_iterations 1 \
--artifact_threshold 0.01 \
--draw_rectangles \
--merge_zones
```

Then, to **transfer the project to VIA2**, please :

- [Download VIA2](https://www.robots.ox.ac.uk/~vgg/software/via/) 
- Open a new VIA2 project and import your images
- In the `project`-menu, chose import file/region attributes and import `default_via_attributes.json` from
the annotation_helper/data directory. 
- In the `annotation`-menu, chose import annotations from csv, and import the annotations you want from the 
annotation_helper output. 

# Run evaluator

Run `python3 evaluator` with the following arguments : 

```shell script
usage: evaluator [-h] [--IMG_DIR IMG_DIR] [--SVG_DIR SVG_DIR]
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
  --evaluation_level EVALUATION_LEVEL
                        The level at which evaluation should be performed
                        ("line" or "word"). Line is not implemented yet !
```

For example : 
```
python evaluator --IMG_DIR "data/test_png" \
--SVG_DIR "data/test_svg" \
--OUTPUT_DIR "output" \
--OCR_DIR "test/"
--GROUNDTRUTH_DIR "test_gt/"
--draw_rectangles \
```