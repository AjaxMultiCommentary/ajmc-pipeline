`region_detection` automatically detects regions-boxes in image-files and outputs 
a csv that can be directly imported to VIA. 

**Detecting boxes in image-files** is done using `cv2.dilation`. This dilates recognized letters-contours to recognize 
wider structures. The retrieved rectangles are then shrinked back to their original size. This can be seen
when drawing rectangles on images. ``

# Usage

⚠️ **CLI usage is not implemented yet** ⚠️

Run `python region_detection` with the following arguments : 

```
usage: region_detection [-h] [--IMG_DIR IMG_DIR] [--SVG_DIR SVG_DIR]
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

For instance : 
```shell script
python region_detection --img_dir "data/test_png" \
--output_dir "output" \
--dilation_kernel_size 51 \
--dilation_iterations 1 \
--artifact_threshold 0.01 \
--draw_rectangles
```

# Synergy with VIA

Then, to **transfer the project to VIA2**, please :

- [Download VIA2](https://www.robots.ox.ac.uk/~vgg/software/via/) 
- Open a new VIA2 project and import your images
- In the `project`-menu, chose import file/region attributes and import `default_via_attributes.json` from
the annotation_helper/data directory. 
- In the `annotation`-menu, chose import annotations from csv, and import the annotations you want from the 
annotation_helper output. 



