# 2022 Spring DHLAB Semester Project - Pushing the limits of optical character recognition on complex multilingual documents

Student: Ziqi Zhao

Supervisor: Sven Najem-Meyer

Term: 2022 Spring

## Main Research Results
This project aims to improve the performance of Tesseract on ancient Greek commentaries. specifically, this project focus on obtaining better Tesseract Greek models and finding proper image pre-processing pipelines for commentaries. By properly choosing the combination of Tesseract language models, the character error rate(CER) will decrease. After fine-tuning or re-training, the obtained models can outperform the original Greek model in the OCR. Some image pre-processing operations can also applied to further improve the performance. Details can be found in the report PDF.

## Setup

First install Tesseract with Tesstrain support from the instructions on [this link.](https://tesseract-ocr.github.io/tessdoc/Compiling-–-GitInstallation.html) A simple starting commands on the ICCluster (without root) are the following:

```bash
# first download leptonica release file, then unzip
gunzip leptonica-1.82.0.tar.gz
tar -xvf leptonica-1.82.0.tar
cd leptonica-1.82.0
./configure --prefix=/some/dir/local
make
make install

# if missing curl, can use the following
# first download curl release file and unzip, then do the following
cd curl
./configure --prefix=/some/dir/local
make
make install

# if missing bc, can follow the instruction on https://github.com/gavinhoward/bc to install it

# then install tesseract
cd tesseract
./autogen.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/some/dir/local/lib
LIBLEPT_HEADERSDIR=/some/dir/local/include ./configure \
               --prefix=/some/dir/local/ \
               --with-extra-libraries=/some/dir/local/lib \
CXXFLAGS="-I/some/dir/local/include" \
LDFLAGS="-L/some/dir/local/lib" \
CFLAGS="-I/some/dir/local/include"

make
make install

# point to the parent folder of tessdata
export TESSDATA_PREFIX=/some/path/to/tessdata

make training
make training-install
```

You can add the export to '~./profile' to permanently solve the path problem:

```bash
export LD_LIBRARY_PATH=/usr/local/lib:/path/to/libraries

export PATH=$PATH:/path/to/libraries

export TESSDATA_PREFIX=/path/to/parent/of/tessdata
```

There are also some support pages: 

https://codingvision.net/build-tesseract-5-in-conda-environment

https://tesseract-ocr.github.io

Download [Best Tessdata](https://github.com/tesseract-ocr/tessdata_best), [Tessconfig](https://github.com/tesseract-ocr/tessconfigs), [Tesstrain](https://github.com/tesseract-ocr/tesstrain) and [Langdata](https://github.com/tesseract-ocr/langdata).

For ease of use, install PyTesseract, skimage and openCV.

## Tmux

If you are using Tmux for your project, don't forget to add '-u' to the command to correctly show the Greek characters in the terminals: 

```bash
tmux -u new -s session-name
```

You can also use the following alias in '~./profile':

```bash
alias tmux='tmux -u'
```

## Code Structure

The code contributes as part of the bigger project Ajax. The implemented code for this project only lies in ajmc/ajmc/ocr. The Tesseract-related folders are put in 'ts', and the PoGreTra dataset should also be in the same folder as 'ajmc' and 'ts':

```
.
├── ajmc/
│   ├── ajmc/
│   │   ├── commons
│   │   ├── nlp
│   │   ├── ocr/
│   │   │   ├── evaluation
│   │   │   ├── exps/
│   │   │   │   ├── evaluation/
│   │   │   │   │   └── commentary1/
│   │   │   │   │       ├── images/png
│   │   │   │   │       ├── ocr/
│   │   │   │   │       │   ├── groundtruth/evaluation
│   │   │   │   │       │   └── runs/
│   │   │   │   │       │       └── ...
│   │   │   │   │       └── olr/
│   │   │   │   │           └── via_project.json
│   │   │   │   └── ...
│   │   │   ├── preprocess/
│   │   │   │   ├── align_image_heights.py
│   │   │   │   ├── clean_dataset.py
│   │   │   │   ├── merge_dataset.py
│   │   │   │   └── toolbox.py
│   │   │   ├── run/
│   │   │   │   ├── aggregate_results.py
│   │   │   │   ├── eval_preprocess.py
│   │   │   │   ├── run_exp_finetune_epoch_new.py
│   │   │   │   ├── run_exp_finetune_height.py
│   │   │   │   ├── run_exp_finetunw_ite_new.py
│   │   │   │   ├── run_exp_retrain_epoch_new.py
│   │   │   │   ├── run_tesseract.py
│   │   │   │   ├── test_preprocess.ipynb
│   │   │   │   └── ...
│   │   │   ├── tesstrain_configs/
│   │   │   │   └── ...
│   │   │   └── README.md
│   │   ├── olr
│   │   └── text_importation
│   ├── examples
│   └── tests
├── pogretra-v1.0/
│   └── ...
└── ts/
    ├── langdata
    ├── tessconfigs
    ├── tessdata_best
    ├── tesseract
    └── tesstrain
```

ajmc/ajmc/ocr/evaluation: evaluation code

ajmc/ajmc/ocr/exps: folder for storing the experiment results.

ajmc/ajmc/ocr/preprocess: folder containing different preprocessing operations for the dataset and the images.

- align_image_heights.py: preprocess the dataset, so that resulting dataset only contains images with a target height
- clean_dataset.py: remove unwanted images based on their languages. Multiple methods are implemented.
- merge_dataset.py: once we have the cleaned dataset, run this to merge the dataset into one folder. Otherwise, Tesseract may not be able to recognize them in the training.
- toolbox.py: some basic image pre-processing steps based on skimage.

ajmc/ajmc/ocr/run: containing the core API for the project, and also the experiment scripts

- aggregate_results.py: generate an aggregated result for each commentary from results in 'ajmc/ajmc/ocr/exps/evaluation/'
- eval_preprocess.py: evaluate the image pre-processing pipeline on the selected models
- run_exp_finetune_epoch_new.py: run the fine-tuning experiments on different fine-tuning epochs.
- run_exp_finetune_height.py: run the fine-tuning experiments on different image heights
- run_exp_finetune_ite_new.py: run the fine-tuning experiments on different iterations.
- run_exp_retrain_epoch_new.py: run the re-training experiments on different re-training epochs.
- run_tesseract.py: contains many useful APIs for IO, training and evaluation.
- test_preprocess.ipynb: an intuitive notebook for testing the functions and visualizing the results.

ajmc/ajmc/ocr/tesstrain_configs: containing all the Tesstrain config files for fine-tuning and re-training.

## API usage
For the API, please refer to test_preprocess.ipynb and the experiment files (run_exp_***.py) for examples. Also refer to the comments in run_tesseract.py.

## Obtained models and full experiment results
The traineddata for obtained models can be found [here](https://drive.google.com/drive/folders/1OYREcMcw5AdCXi8DhlFNAMH7a23owuwF?usp=sharing). The complete results can also be found in this shared drive.