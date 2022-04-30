import os
import cv2
import sys
import yaml
import shutil
import pytesseract
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats, clear_output
set_matplotlib_formats('svg')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

PROJECT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
PARENT_DIR = os.path.dirname(PROJECT_DIR)

sys.path.append(os.path.join(PROJECT_DIR, "commons"))
import variables

# Data dir with whole images
RAW_DATA_DIR = variables.PATHS.get("base_dir", "")
RAW_COMMENTARY_DIRS = ['Wecklein1894', 'Kamerbeek1953', 'sophoclesplaysa05campgoog', 'Paduano1982', 'lestragdiesdeso00tourgoog', 
                        'Untersteiner1934', 'Ferrari1974', 'sophokle1v3soph', 'DeRomilly1976', 'Finglass2011', 'Colonna1975', 
                        'bsb10234118', 'cu31924087948174', 'Garvie1998']

# Data dir containing line images and ground truths
TRAIN_DATA_DIR = os.path.join(PARENT_DIR, "GT-commentaries-OCR", "data")
TRAIN_COMMENTARY_DIRS = ['Wecklein1894', 'sophoclesplaysa05campgoog', 'bsb10234118', 'sophokle1v3soph', 'cu31924087948174']

POGRETRA_DATA_DIR = os.path.join(PARENT_DIR, "pogretra-v1.0", "Data")
POGRETRA_COMMENTARY_DIRS = ['German-serifs/ldpd_10922736_000', 'German-serifs/OU_ligature_ST_ligature/nnc1.50178386-1593613693-redux', 
                        'German-serifs/platonisdialogi06plat', 'German-serifs/actaphilippietac00bonnuoft', 'German-serifs/bsb10234118', 
                        'German-serifs/stoicorumveterum02arniuoft', 'German-serifs/bub_gb_FZbfaq7tcvAC', 'German-serifs/churchfathers', 
                        'German-serifs/b21459162_0003', 'Porson/Porson-commentaries', 'Porson/Old-Oxford', 'Porson/Porson', 
                        'Porson/sourcesforgreek02hillgoog', 'Old-Teubner-serif/602250676brucerob', 
                        'Old-Teubner-serif/inplatonisrempu02krolgoog_teubner_serif', 'Old-Teubner-serif/aeschinisoration00aesc', 
                        'Old-Teubner-serif/Teubner-serif-training', 'Old-Teubner-serif/sextiempiriciope12sext07', 
                        'Old-Teubner-serif/deanimaliumantur02aeliuoft', 'Old-Teubner-serif/poetaeminoresgra02gais']

TESSDATA_DIR = os.path.join(PARENT_DIR, 'ts', 'tessdata')
TESSDATA_BEST_DIR = os.path.join(PARENT_DIR, 'ts', 'tessdata_best')

def get_commentary_dir(commentary_name, mode="train", cleaned=False, create_if_missing=False):
    '''
        Get the dir name from given arguments

        params:
            - commentary_name: the name of folder for one commentary
            - mode: the organization of the dataset. 'train' means that it has a folder called 'GT-pairs' and 
                    it includes line images and their ground truth; 'raw' means that it has a folder called 
                    'images/png' and it only includes images for whole pages and there is no ground truth
            - cleaned: whether the dataset has been cleaned by function 'clean_data_tesstrain'. If set to True,
                    then the resulting folder will have a suffix '-clean'
            - create_if_missing: when there is no such folder, set this to True to allow the code to create it,
                    otherwise it will raise a FileNotFoundError
    '''
    assert(mode in ["train", "raw", "pogretra"])
    if mode == "train":
        commentary_dir = os.path.join(TRAIN_DATA_DIR, commentary_name, "GT-pairs-clean" if cleaned else "GT-pairs")
    elif mode == "raw":
        commentary_dir = os.path.join(RAW_DATA_DIR, commentary_name, "images", "png-clean" if cleaned else "png")
    else:
        commentary_dir = os.path.join(POGRETRA_DATA_DIR, f"{commentary_name}-clean" if cleaned else commentary_name)
    if not os.path.isdir(commentary_dir):
        if create_if_missing:
            print(f"creating dir: {commentary_dir}")
            os.makedirs(commentary_dir)
        else:
            raise FileNotFoundError(f"folder doesn't exist: {commentary_dir}. Check your folder settings, or enable 'create_if_missing' in function get_commentary_dir.")
    return commentary_dir

def get_fig_name(commentary_name, fig_name, mode="train", cleaned=False):
    '''
        Get the path of a figure given the arguments

        params:
            - commentary_name: the name of folder for one commentary
            - fig_name: the index of the figure. Note that the figures' filenames are organized as
                    <commentary_name>_<fig_name>.
            - mode: the organization of the dataset. 'train' means that it has a folder called 'GT-pairs' and 
                    it includes line images and their ground truth; 'raw' means that it has a folder called 
                    'images/png' and it only includes images for whole pages and there is no ground truth
            - cleaned: whether the dataset has been cleaned by function 'clean_data_tesstrain'. If set to True,
                    then the resulting folder will have a suffix '-clean'
    '''
    commentary_dir = get_commentary_dir(commentary_name, mode, cleaned)
    fig = os.path.join(commentary_dir, f"{commentary_name}_{fig_name}") if mode in ["raw", "train"] else os.path.join(commentary_dir, fig_name)
    assert(os.path.isfile(fig))
    return fig

def get_fig_idxs(commentary_name, mode="train", cleaned=False, verbose=False):
    '''
        Get a list of figures' indices from a given dataset

        params:
            - commentary_name: the name of folder for one commentary
            - mode: the organization of the dataset. 'train' means that it has a folder called 'GT-pairs' and 
                    it includes line images and their ground truth; 'raw' means that it has a folder called 
                    'images/png' and it only includes images for whole pages and there is no ground truth
            - cleaned: whether the dataset has been cleaned by function 'clean_data_tesstrain'. If set to True,
                    then the resulting folder will have a suffix '-clean'
            - verbose: set to True to print the fig_list and the length of it
    '''
    commentary_dir = get_commentary_dir(commentary_name, mode, cleaned)
    init_fig_list = [item for item in list(os.listdir(commentary_dir)) if item.endswith("png")]
    fig_list = sorted([item.replace(f"{commentary_name}_", "") if mode in ["raw", "train"] else item for item in init_fig_list])
    if verbose:
        print(f"Total img count: {len(fig_list)}")
        if len(fig_list) >= 200:
            print("too many images. only showing the first 100 and last 100.")
            print(fig_list[0:100])
            print(fig_list[-100:])
        else:
            print(fig_list)
    return fig_list

def show_fig(commentary_name, fig_name, mode="train", cleaned=False):
    '''
        Show an image in the screen.
    '''
    fig = get_fig_name(commentary_name, fig_name, mode, cleaned)
    img = cv2.imread(fig)
    plt.imshow(img)
    plt.axis("off")
    return img

def clean_data_tesstrain(commentary_name, mode="train", cleaned=False):
    '''
        Remove .box and .lstmf files from the dataset. These files are created by tesstrain.
    '''
    commentary_dir = get_commentary_dir(commentary_name, mode, cleaned)
    count = 0
    for file in os.listdir(commentary_dir):
        if file.endswith(".box") or file.endswith(".lstmf"):
            print(f"removing {file}")
            os.remove(os.path.join(commentary_dir, file))
            count += 1
    print(f"In total remove {count} files.")

def check_missing_gt(commentary_name, mode="train", cleaned=False):
    '''
        Given a dataset, check whether there is any bad ground truth files (missing or empty).
    '''
    commentary_dir = get_commentary_dir(commentary_name, mode, cleaned)
    missing = []
    for file in sorted(list(os.listdir(commentary_dir))):
        if file.endswith(".png"):
            gt_file = os.path.join(commentary_dir, file.replace(".png", ".gt.txt"))
            if os.path.isfile(gt_file) and os.path.getsize(gt_file) > 0:
                continue
            else:
                print(f"Empty ground truth file or missing file: {gt_file}")
                missing.append(file)
    print(f"In total {len(missing)} bad ground truth files.")
    return missing

def clean_gt_folder(commentary_name, mode="train"):
    ''''
        Given a dataset, create a clean version and exclude all bad ground truth files (missing or empty)
    '''
    commentary_dir = get_commentary_dir(commentary_name, mode)
    clean_commentary_dir = get_commentary_dir(commentary_name, mode, cleaned=True, create_if_missing=True)
    missing = check_missing_gt(commentary_name, mode, cleaned=False)
    count = 0
    for filename in os.listdir(commentary_dir):
        if filename.endswith(".png") and filename not in missing and os.path.isfile(os.path.join(commentary_dir, filename.replace(".png", ".gt.txt"))):
            shutil.copy(os.path.join(commentary_dir, filename), os.path.join(clean_commentary_dir, filename))
            shutil.copy(os.path.join(commentary_dir, filename.replace(".png", ".gt.txt")), os.path.join(clean_commentary_dir, filename.replace(".png", ".gt.txt")))
            count += 1
    print(f"Copy {count} GT-pairs. Skip {len(missing)} GT-pairs because they have bad ground truth file.")

def test_ocr(exp_name, tessdata_dir, commentary_name, fig_idx, lang="eng+fra+grc", mode="train", save=False, viz=True, verbose=True, add_timestamp=True, cleaned=False):
    '''
        Use Tesseract to recognize a given image, save the relevant output and visualize the OCR result.

        params:
            - exp_name: the output folder. The folder will appear in ocr/exps. If add_timestamp is set to
                    True, then a timestamp will be added to the folder name
            - tessdata_dir: the folder to store the tessdata. Equivelant to use TESSDATA_PREFIX.
            - commentary_name: the name of folder for one commentary
            - fig_idx: the index of figure. Note that the figures' filenames are organized as
                    <commentary_name>_<fig_name>.
            - lang: language models to use. Use '+' to concatenate multiple language models
            - mode: the organization of the dataset. 'train' means that it has a folder called 'GT-pairs' and 
                    it includes line images and their ground truth; 'raw' means that it has a folder called 
                    'images/png' and it only includes images for whole pages and there is no ground truth
            - save: whether or not to save the OCR results to the folder.
            - viz: whether or not to visualize the OCR results.
            - verbose: whether or not to print the messages
            - add_timestamp: If add_timestamp is set to True, then a timestamp will be added to the folder name
            - cleaned: whether the dataset has been cleaned by function 'clean_data_tesstrain'. If set to True,
                    then the resulting folder will have a suffix '-clean'
            
    '''
    fig_name = get_fig_name(commentary_name, fig_idx, mode, cleaned)

    img = cv2.imread(fig_name)

    return test_ocr_raw(exp_name, tessdata_dir, img, fig_name, lang=lang, save=save, viz=viz, verbose=verbose, add_timestamp=add_timestamp)

def test_ocr_raw(exp_name, tessdata_dir, img, output_name, lang="eng+fra+grc", save=False, viz=True, verbose=True, add_timestamp=True):
    '''
        Use Tesseract to recognize a given image, save the relevant output and visualize the OCR result.

        params:
            - exp_name: the output folder. The folder will appear in ocr/exps. If add_timestamp is set to
                    True, then a timestamp will be added to the folder name
            - tessdata_dir: the folder to store the tessdata. Equivelant to use TESSDATA_PREFIX.
            - img: the opened img file.
            - output_name: the output prefix for OCR results.
            - lang: language models to use. Use '+' to concatenate multiple language models
            - mode: the organization of the dataset. 'train' means that it has a folder called 'GT-pairs' and 
                    it includes line images and their ground truth; 'raw' means that it has a folder called 
                    'images/png' and it only includes images for whole pages and there is no ground truth
            - save: whether or not to save the OCR results to the folder.
            - viz: whether or not to visualize the OCR results.
            - verbose: whether or not to print the messages
            - add_timestamp: If add_timestamp is set to True, then a timestamp will be added to the folder name
            - cleaned: whether the dataset has been cleaned by function 'clean_data_tesstrain'. If set to True,
                    then the resulting folder will have a suffix '-clean'
            
    '''

    if add_timestamp:
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        output_folder = os.path.join(PROJECT_DIR, "ocr", "exps", f"{timestamp}_{exp_name}")
    else:
        output_folder = os.path.join(PROJECT_DIR, "ocr", "exps", exp_name)

    custom_config = f"--tessdata-dir {tessdata_dir} -l {lang} --oem 1 -c lstm_choice_mode=2"
    output = pytesseract.image_to_string(img, config=custom_config)
    output_hocr = pytesseract.image_to_pdf_or_hocr(img, extension="hocr", config=custom_config).decode("utf-8")

    if save:
        os.makedirs(output_folder, exist_ok=True)

        output_file = os.path.join(output_folder, os.path.basename(output_name)+".str")
        output_file_hocr = os.path.join(output_folder, os.path.basename(output_name)+".hocr")

        with open(output_file, "w", encoding="utf-8") as f_out:
            f_out.write(output)
            print(f"string output saved to {output_file}") if verbose else None

        with open(output_file_hocr, "w", encoding="utf-8") as f_out:
            f_out.write(output_hocr)
            print(f"hocr output saved to {output_file_hocr}") if verbose else None

    if viz:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(img)
        axs[0].axis("off")
        axs[1].text(0,0,output, fontsize=3)
        axs[1].axis("off")
        plt.show(fig)

    print(f"OCR output:\n{output}") if verbose else None
    return output

def batch_ocr(exp_name, tessdata_dir, commentary_name, lang="eng+fra+grc", mode="train", save=True, viz=False, verbose=False, cleaned=False):
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    exp_name = f"{timestamp}_{exp_name}"

    for fig_idx in tqdm(get_fig_idxs(commentary_name, mode, cleaned)):
        output = test_ocr(exp_name, tessdata_dir, commentary_name, fig_idx, lang, mode, save, viz, verbose, add_timestamp=False, cleaned=cleaned)
        

def train(model_name, commentary_names, mode, output_dir, cleaned=True, config_file=None):
    TESSTRAIN_DIR = os.path.join(PARENT_DIR, "ts", "tesstrain")
    sh_file = os.path.join(TESSTRAIN_DIR, "tmp_train.sh")

    def log_str(cmd, file):
        return f"{cmd} 2>&1 | tee -a {file} >/dev/full; \n"

    def log(msg, file):
        with open(file, "a+") as tmp_file:
            tmp_file.write(msg)

    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log_file = os.path.join(TESSTRAIN_DIR, f"log-{timestamp}.txt")

    print(f"See {log_file} for the training output.")

    if config_file:
        with open(config_file, "r") as f_in:
            configs = yaml.full_load(f_in)
    else:
        configs = {}
    log(str(configs), log_file)

    cmd_line = f"make training MODEL_NAME={model_name} DATA_DIR={output_dir} "

    for commentary_name in commentary_names:

        commentary_dir = get_commentary_dir(commentary_name, mode=mode, cleaned=cleaned)
    
        cmd_line += f"GROUND_TRUTH_DIR={commentary_dir} "

    
    for field in configs:
        if field in ["MODEL_NAME", "DATA_DIR, GROUND_TRUTH_DIR"]:
            log(f"Skip {field} in yaml, because the function already has such variable.", log_file)
            continue
        if configs[field] in ["", "none"]:
            log(f"Skip empty field '{field}'.", log_file)
            continue
        cmd_line += f" {field}={configs[field]}"
    log(cmd_line,log_file)
    print(cmd_line)

    with open(sh_file, "w") as f_out:
        f_out.write(f"cd {TESSTRAIN_DIR};\n")

        f_out.write(log_str(cmd_line, log_file))
    os.system(f"sh {sh_file}")