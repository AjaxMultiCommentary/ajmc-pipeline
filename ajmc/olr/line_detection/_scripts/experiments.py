#%% Imports and functions declarations
import json
import shutil
from pathlib import Path
from typing import List, Dict

import cv2

from ajmc.commons import geometry as geom, variables as vs
from ajmc.commons.arithmetic import safe_divide
from ajmc.commons.image import draw_box
from ajmc.commons.miscellaneous import get_ajmc_logger, ROOT_LOGGER
from ajmc.olr.evaluation import compute_shapes_confusion_matrix, compute_mean_iou
from ajmc.olr.line_detection import models
from ajmc.olr.line_detection.data_processing import get_pages_lines
from ajmc.text_processing.raw_classes import RawCommentary

ROOT_LOGGER.setLevel('INFO')
logger = get_ajmc_logger(__name__)


def compute_micro_averaged_iou(predictions: Dict[str, List[geom.Shape]], groundtruth: Dict[str, List[geom.Shape]]) -> float:
    """Compute the micro-averaged IoU score for the given predictions and groundtruth.
    
    Note:
        The micro-averaged IoU score is the mean IoU score for each page.
    
    Args:
        predictions: The predictions for each page, as ``{"page_id": [line1, line2, ...]}``.
        groundtruth: The groundtruth for each page, as ``{"page_id": [line1, line2, ...]}``.
    """

    scores = [compute_mean_iou(lines, groundtruth[page_id]) for page_id, lines in predictions.items()]

    return sum(scores) / len(scores)


def compute_micro_averaged_metrics(predictions: Dict[str, List[geom.Shape]], groundtruth: Dict[str, List[geom.Shape]]) -> Dict[str, float]:
    # Create the confusion matrix for each model

    scores = {k: [] for k in ['F1', 'Recall', 'Precision']}

    for page_id, lines in predictions.items():
        conf_mtrx = compute_shapes_confusion_matrix(lines, groundtruth[page_id])
        scores['F1'].append(2 * conf_mtrx['TP'] / (2 * conf_mtrx['TP'] + conf_mtrx['FP'] + conf_mtrx['FN']))
        scores['Recall'].append(conf_mtrx['TP'] / (conf_mtrx['TP'] + conf_mtrx['FN']))
        scores['Precision'].append(safe_divide(conf_mtrx['TP'], (conf_mtrx['TP'] + conf_mtrx['FP'])))

    return {k: sum(v) / len(v) for k, v in scores.items()}


def import_predictions(page_ids: List[str]):
    predictions = {'blla': {}, 'legacy': {}, 'blla_adjusted': {}, 'legacy_adjusted': {}}

    comm_ids = set([page_id.split('_')[0] for page_id in page_ids])
    for comm_id in comm_ids:
        comm_olr_dir = vs.get_comm_olr_lines_dir(comm_id)
        comm_page_ids = [page_id for page_id in page_ids if page_id.startswith(comm_id)]

        for page_id in comm_page_ids:
            for model_name in predictions.keys():
                page_path = comm_olr_dir / model_name / (page_id + '.json')
                page_preds = json.loads(page_path.read_text())
                predictions[model_name][page_id] = [geom.Shape(points) for points in page_preds]

    predictions['easy_ocr'] = {
        page_id: [geom.Shape(points) for points in json.loads((EXP_DIR / 'outputs/easy_ocr' / (page_id + '.json')).read_text())] for page_id in
        page_ids}
    predictions['tesseract'] = {}

    for comm_id in comm_ids:
        comm = RawCommentary(comm_id, '*_tess_retrained')
        comm_page_ids = [page_id for page_id in page_ids if page_id.startswith(comm_id)]
        for page in comm.children.pages:
            if page.id in comm_page_ids:
                predictions['tesseract'][page.id] = [l.bbox for l in page.children.lines]

    return predictions


def write_easy_ocr_preds():
    easy_ocr_model = models.EasyOCRModel()
    images_dir = EXP_DIR / 'images'
    images_dir.mkdir(exist_ok=True)

    output_dir = EXP_DIR / 'outputs/easy_ocr'
    output_dir.mkdir(exist_ok=True)

    all_preds = {}
    for page_id in VIA['_via_img_metadata'].keys():
        comm_id = page_id.split('_')[0]
        img_path = vs.get_comm_img_dir(comm_id) / (page_id + '.png')
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_path = images_dir / (page_id + '.png')
        cv2.imwrite(str(img_path), img)
        output_path = output_dir / (page_id + '.json')
        page_preds = easy_ocr_model.predict(img_path)
        page_preds_json = [[[int(s.xyxy[0]), int(s.xyxy[1])], [int(s.xyxy[2]), int(s.xyxy[3])]] for s in page_preds]
        output_path.write_text(json.dumps(page_preds_json))
        all_preds[page_id] = easy_ocr_model.predict(img_path)


# Set directories and paths

EXP_DIR = Path('/Users/sven/Desktop/line_detection_experiments')
EXP_DIR.mkdir(exist_ok=True)
LINES_VIA_PATH = Path('/Users/sven/packages/ajmc_data/GT-commentaries-lines/GT-commentaries-lines-balanced.json')
VIA = json.loads(LINES_VIA_PATH.read_text(encoding='utf-8'))

#%% Import the ground truth lines and the predictions
balanced_groundtruth = get_pages_lines(VIA)
predictions = import_predictions(list(balanced_groundtruth.keys()))

#%% Get the results of the baselines

results = {}

for model_name in ['blla', 'legacy', 'blla_adjusted', 'legacy_adjusted', 'easy_ocr', 'tesseract']:
    # Declare the model's results dict, setting all the parameters to None (as baselines do not have parameters)
    results[model_name] = {k: None for k in ['split_lines', 'double_line_threshold', 'minimal_height_factor', 'line_inclusion_threshold']}
    results[model_name]['iou'] = compute_micro_averaged_iou(predictions[model_name], balanced_groundtruth)
    scores = compute_micro_averaged_metrics(predictions[model_name], balanced_groundtruth)
    for metric, score in scores.items():
        results[model_name][metric] = score

#%% Create a dataframe with the results of the preliminary test
import pandas as pd

pretest_results = pd.DataFrame(results).T
# Drop the first 4 columns as they are all None
pretest_results = pretest_results.drop(columns=['split_lines', 'double_line_threshold', 'minimal_height_factor', 'line_inclusion_threshold'])

# Drop the blla_adjusted and legacy_adjusted rows as they are the same as the blla and legacy columns
pretest_results = pretest_results.drop(index=['blla_adjusted', 'legacy_adjusted'])

#Rename the index
pretest_results = pretest_results.rename(index={'blla': 'KrakenBlla', 'legacy': 'KrakenLegacy', 'easy_ocr': 'EasyOCR', 'tesseract': 'Tesseract'})

# Rename the column iou to IoU
pretest_results = pretest_results.rename(columns={'iou': 'IoU'})

styler = pretest_results.style

# Export this dataframe to latex
styler.format(escape='latex', precision=3)

# Highlight the best results in each column
styler.highlight_max(axis=0, props='font-weight: bold')

styler.set_caption("""Results of the preliminary tests on the balanced dataset.""")

styler.set_table_styles()

latex = styler.to_latex(hrules=True,
                        position_float='centering',
                        convert_css=True,
                        label=f'tab:4_1 {styler.caption}')

#%% Draw the outputs of the models in the preliminary test

COLORS = {'groundtruth': vs.COLORS['distinct']['green'],
          'blla': vs.COLORS['distinct']['red'],
          'legacy': vs.COLORS['distinct']['blue'],
          'easy_ocr': vs.COLORS['distinct']['purple'],
          'tesseract': vs.COLORS['distinct']['ecru']}

output_dir = EXP_DIR / 'pretests_outputs'
shutil.rmtree(output_dir)
output_dir.mkdir(exist_ok=True)

sample_pages_ids = ['cu31924087948174_0086',
                    'sophoclesplaysa05campgoog_0014',
                    'sophoclesplaysa05campgoog_0204']

for page_id in sample_pages_ids:

    img_path = vs.get_comm_img_dir(page_id.split('_')[0]) / (page_id + '.png')

    for model_name in ['blla', 'legacy', 'easy_ocr', 'tesseract']:
        img = cv2.imread(str(img_path))
        output_path = output_dir / (page_id + f'_{model_name}.png')
        lines = predictions[model_name][page_id]

        for line in lines:
            img = draw_box(line.bbox, img, stroke_color=vs.COLORS['hues']['dodger_blue'], stroke_thickness=max(2, int(img.shape[1] / 1000)))

        cv2.imwrite(str(output_path), img)

#%% Draw the outputs of GT

for page_id in sample_pages_ids:

    img_path = vs.get_comm_img_dir(page_id.split('_')[0]) / (page_id + '.png')
    output_path = output_dir / f'{page_id}_GT.png'
    img = cv2.imread(str(img_path))
    lines = balanced_groundtruth[page_id]

    for line in lines:
        img = draw_box(line.bbox, img, stroke_color=COLORS['groundtruth'], stroke_thickness=max(2, int(img.shape[1] / 1000)))

    cv2.imwrite(str(output_path), img)

#%% Compute the experiments (run only once)

for split_lines in [True]:
    for double_line_threshold in [1.4, 1.6]:
        for minimal_height_factor in [0.4, 0.35, 0.5]:
            for line_inclusion_threshold in [0.6, 0.65]:

                combined_model = models.CombinedModel(adjusted_legacy_model=None,
                                                      adjusted_blla_model=None,
                                                      line_inclusion_threshold=line_inclusion_threshold,
                                                      minimal_height_factor=minimal_height_factor,
                                                      double_line_threshold=double_line_threshold,
                                                      split_lines=split_lines)

                model_name = f'combined_dl{double_line_threshold}_mh{minimal_height_factor}_it{line_inclusion_threshold}_sp{int(split_lines)}'
                if model_name in predictions:
                    continue
                print(f'Running {model_name}...')

                # COMPUTE THE COMBINED LINES
                predictions[model_name] = {}
                for page_id in balanced_groundtruth.keys():
                    predictions[model_name][page_id] = combined_model.predict(None,
                                                                              legacy_predictions=predictions['legacy_adjusted'][page_id],
                                                                              blla_predictions=predictions['blla_adjusted'][page_id])

                # Evaluate the predictions
                results[model_name] = {'split_lines': split_lines,
                                       'double_line_threshold': double_line_threshold,
                                       'minimal_height_factor': minimal_height_factor,
                                       'line_inclusion_threshold': line_inclusion_threshold}

                results[model_name]['iou'] = compute_micro_averaged_iou(predictions[model_name], balanced_groundtruth)
                scores = compute_micro_averaged_metrics(predictions[model_name], balanced_groundtruth)
                for metric, score in scores.items():
                    results[model_name][metric] = score

                # Draw the bounding boxes

                # Create the output directory
                output_dir = EXP_DIR / f'outputs/{model_name}'
                shutil.rmtree(output_dir, ignore_errors=True)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Prepare the esthetics
                colors = {'legacy_adjusted': vs.COLORS['distinct']['blue'],
                          'blla_adjusted': vs.COLORS['distinct']['green'],
                          model_name: vs.COLORS['distinct']['red']}

                thicknesses = {'legacy_adjusted': 1,
                               'blla_adjusted': 1,
                               model_name: 2}

                # We now draw the bounding boxes of the predictions and the ground truth
                for page_id, lines in predictions[model_name].items():
                    img_path = vs.get_comm_img_dir(page_id.split('_')[0]) / (page_id + '.png')
                    img = cv2.imread(str(img_path))
                    output_path = output_dir / (page_id + '.png')

                    for model_name_ in ['legacy_adjusted', 'blla_adjusted', model_name]:
                        for line in predictions[model_name_][page_id]:
                            img = draw_box(line.bbox, img, stroke_color=colors[model_name_],
                                           stroke_thickness=thicknesses[model_name_])

                    cv2.imwrite(str(output_path), img)

                # Write files
                conf = {'double_line_threshold': double_line_threshold,
                        'minimal_height_factor': minimal_height_factor,
                        'line_inclusion_threshold': line_inclusion_threshold,
                        'split_lines': split_lines}

                (output_dir / 'predictions.json').write_text(
                        json.dumps([p.points for preds in predictions[model_name].values() for p in preds], indent=2))
                (output_dir / 'config.json').write_text(json.dumps(conf, indent=2))
                (output_dir / 'results.json').write_text(json.dumps(results[model_name], indent=2))
                (output_dir.parent / 'all_results.json').write_text(json.dumps(results, indent=2))

#%% Inspect the results and export them to latex


import pandas as pd

try:
    df = pd.DataFrame.from_dict(results, orient='index')
except NameError:
    results = json.loads((EXP_DIR / 'outputs/all_results.json').read_text(encoding='utf-8'))
    df = pd.DataFrame.from_dict(results, orient='index')

# Keep only the rows where line_inclusion_threshold is either 0.65 or None
df = df[(df['line_inclusion_threshold'] == 0.65) | (df['line_inclusion_threshold'].isna())]

# Drop the split_lines and the line_inclusion_threshold column
df = df.drop(columns=['split_lines', 'line_inclusion_threshold'])

# Rename the columns
df = df.rename(columns={'double_line_threshold': '\(h_{max}\)',
                        'minimal_height_factor': '\(h_{min}\)',
                        'iou': 'IoU', })

# Rename the index
# df = df.rename(index=lambda x: 'Combined' if 'combined' in x else x)
df = df.rename(index={'blla': 'Blla', 'legacy': 'Legacy',
                      'legacy_adjusted': 'LegacyAdjusted',
                      'blla_adjusted': 'BllaAdjusted'})

# Name the index
df.index.name = 'Model'

# Sort the rows by max height, then by min height, so that nan values are at the begining
df = df.sort_values(by=['\(h_{max}\)', '\(h_{min}\)'], na_position='first')

#%%
styler = df.style

# Export this dataframe to latex
styler.format(escape='latex', na_rep='-')

# Highlight the best results in each column, only for the IoU, Precision, Recall and F1 columns
styler.highlight_max(props='font-weight: bold', axis=0, subset=['IoU', 'Precision', 'Recall', 'F1'])

# Set the precision of the floats, using 1 decimal for the max and min height, and 3 decimals for the rest
styler = styler.format({'\(h_{max}\)': '{:.1f}',
                        '\(h_{min}\)': '{:.2f}',
                        'IoU': '{:.3f}',
                        'Precision': '{:.3f}',
                        'Recall': '{:.3f}',
                        'F1': '{:.3f}'})

styler.set_caption(
    """Results of the line detection experiments. The best results are emboldened column-wise. \(h_{max}\) and \(h_{min}\) respectively represent the maximum and minimum height thresholds used for each combined model.""")

styler.set_table_styles()

latex = styler.to_latex(hrules=True,
                        position_float='centering',
                        convert_css=True,
                        label=f'tab:4_1 {styler.caption.split(".")[0]}')

import re

latex = re.sub(r'combined.*? ', 'Combined ', latex)
latex = re.sub(r'Model &.*?\\\\\n', '', latex)
latex = re.sub(r'\n & \\\(h', r'\nModel & \(h', latex)
latex = latex.replace(' nan ', ' - ')
