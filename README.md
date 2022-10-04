# Presentation

`ajmc` is work-in-progress python package containing the tools developped for 
the [AjaxMultiCommentary project](https://mromanello.github.io/ajax-multi-commentary/).

The project starts from images of classical commentaries and deals with OCR, OLR, and further NLP tasks such as NER and reference resolution.

These steps are covered by the `ajmc`'s sub-packages: 

- `ajmc/commons` contains shared tools and variables.
- `ajmc/nlp` contains helpers for named entity recognition with HuggingFace transformers.
- `ajmc/ocr` contains helpers and functions to run tesseract and perform coordinate-based evaluation of OCR outputs.
- `ajmc/olr` contains helpers and function to perform layout analysis with YOLOv5 and LayoutLM, as well preparing layout annotation with [VIA2](https://www.robots.ox.ac.uk/~vgg/software/via/).
- `ajmc/text_processing` offers a general framework to deal with ocr-output and store them as canonical jsons. 

For a more detailed description of the codebase's architecture and innerworkings, please refer to the `./notebooks` directory. 


# Setup

## Install from source

Please install `ajmc` using `git clone https://github.com/AjaxMultiCommentary/ajmc`. `setup.py` lists all the required dependencies. For collaborative development, please use a dedicated virtual environment (for instance with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)). 




## Install with PIP

```shell
python -m pip install git+'https://github.com/AjaxMultiCommentary/ajmc'
```
