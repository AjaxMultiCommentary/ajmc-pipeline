# Presentation

`ajmc` is work-in-progress python package containing the tools developped for 
the [AjaxMultiCommentary project](https://mromanello.github.io/ajax-multi-commentary/).

The project starts from images of classical commentaries and deals with OCR, OLR, and further NLP tasks. 
These steps are covered by the `ajmc`'s sub-packages. 

- `ocr` notably contains a coordinate-base ocr evaluation tool.
- `olr` notably contains a helper tool to prepare layout annotation with [VIA2](https://www.robots.ox.ac.uk/~vgg/software/via/).
- `text_importation` offers a general framework to deal with ocr-outputs. 


# Setup

## Install from source

Please install `ajmc` using `git clone https://github.com/AjaxMultiCommentary/ajmc`. `requirements.txt` 
specifies the pip-requirements for creating an environment 
(for instance with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)). 



## Install with PIP

```shell
git clone https://github.com/AjaxMultiCommentary/ajmc
cd ajmc
pip install .
```