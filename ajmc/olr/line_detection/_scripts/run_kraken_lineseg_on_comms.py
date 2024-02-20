"""
Run Kraken's line segmentation models on the commentary images. The models are run on the images and the predictions are saved as JSON files in
the commentary's output directory. The models should be run in the following order:
1. The legacy model
2. The legacy model with adjustments
3. The BLLA model
4. The BLLA model with adjustments
5. The combined model
"""

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from ajmc.commons import variables as vs, geometry as geom
from ajmc.commons.image import draw_box
from ajmc.commons.miscellaneous import ROOT_LOGGER, get_ajmc_logger
from ajmc.olr.line_detection import models


parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', type=str,
                    default=['legacy', 'blla', 'legacy_adjusted', 'blla_adjusted', 'combined'],
                    help="""The name of the models to run. The models are run in the order they are given, so please follow this order: """
                         """'legacy', 'blla', 'legacy_adjusted', 'blla_adjusted', 'combined'.""")
parser.add_argument('--blla_model_path', type=str, default='~/opt/anaconda3/envs/kraken/lib/python3.10/site-packages/kraken/blla.mlmodel')

parser.add_argument('--artifact_size_threshold', type=float, default=0.003)
parser.add_argument('--remove_side_margins', type=float, default=0)
parser.add_argument('--line_inclusion_threshold', type=float, default=0.65)
parser.add_argument('--minimal_height_factor', type=float, default=0.3)
parser.add_argument('--double_line_threshold', type=float, default=1.6)
parser.add_argument('--split_lines', action='store_true')

parser.add_argument('--draw_dir', type=str, required=False)

parser.add_argument('--commentary_ids', nargs='+', type=str, default=vs.ALL_COMM_IDS)
parser.add_argument('--logging_level', type=str, default='INFO')
parser.add_argument('--force_recompute', action='store_true')


def run_adjusted_model(seg_type: str, img_path: 'Path'):
    """Run the adjusted model on the image at img_path. If the base model's predictions are available, they are used as input."""
    source_model = seg_type.split('_')[0]
    base_model_predictions_path = vs.get_comm_olr_lines_dir(comm_id) / source_model / f'{img_path.stem}.json'

    if base_model_predictions_path.is_file():
        base_model_predictions = json.loads(base_model_predictions_path.read_text())
        base_model_predictions = [geom.Shape(points) for points in base_model_predictions]
        predictions = all_models[seg_type].predict(img_path=img_path, base_model_predictions=base_model_predictions)

    else:
        predictions = all_models[seg_type].predict(img_path=img_path)

    return predictions


# Prepare the esthetics
colors = {'legacy_adjusted': vs.COLORS['distinct']['blue'],
          'blla_adjusted': vs.COLORS['distinct']['green'],
          'combined': vs.COLORS['distinct']['red']}

thicknesses = {'legacy_adjusted': 1,
               'blla_adjusted': 1,
               'combined': 2}

if __name__ == '__main__':

    # Parse arguments
    args = parser.parse_args()
    if args.draw_dir is not None:
        args.draw_dir = Path(args.draw_dir)
        args.draw_dir.mkdir(exist_ok=True, parents=True)

    # Testing
    # # args.commentary_ids = ['bsb10234118']
    # args.models = [
    #     #     # 'legacy',
    #     'legacy_adjusted',
    #     'blla_adjusted',
    #     'combined'
    # ]
    # # args.remove_side_margins = 0.5
    # args.split_lines = True
    # args.double_line_threshold = 1.4
    # args.minimal_height_factor = 0.4
    # args.line_inclusion_threshold = 0.65
    # args.draw_dir = Path('/Users/sven/Desktop/coucou')
    # args.draw_dir.mkdir(exist_ok=True, parents=True)

    # Set up logging
    ROOT_LOGGER.setLevel(args.logging_level)
    logger = get_ajmc_logger(__name__)

    # Load blla models
    all_models = {}
    all_models['blla'] = models.KrakenBllaModel(args.blla_model_path)
    all_models['blla_adjusted'] = models.AdjustedModel(all_models['blla'], args.artifact_size_threshold, args.remove_side_margins)

    # Load legacy models
    all_models['legacy'] = models.KrakenLegacyModel()
    all_models['legacy_adjusted'] = models.AdjustedModel(all_models['legacy'], args.artifact_size_threshold, args.remove_side_margins)

    # Load combined model
    all_models['combined'] = models.CombinedModel(all_models['legacy_adjusted'],
                                                  all_models['blla_adjusted'],
                                                  line_inclusion_threshold=args.line_inclusion_threshold,
                                                  minimal_height_factor=args.minimal_height_factor,
                                                  double_line_threshold=args.double_line_threshold,
                                                  split_lines=args.split_lines)

    for comm_id in args.commentary_ids:

        # Create the output directories and write configs down
        for model_name in args.models:
            model_output_dir = (vs.get_comm_olr_lines_dir(comm_id) / model_name)
            model_output_dir.mkdir(parents=True, exist_ok=True)

            if 'adjusted' in model_name:
                model_config = {
                    'artifact_size_threshold': args.artifact_size_threshold,
                    'remove_side_margins': args.remove_side_margins
                }
                model_config_path = model_output_dir / 'config.json'
                if not model_config_path.is_file() or args.force_recompute:
                    model_config_path.write_text(json.dumps(model_config, indent=2))

            elif model_name == 'combined':
                model_config = {
                    'line_inclusion_threshold': args.line_inclusion_threshold,
                    'minimal_height_factor': args.minimal_height_factor,
                    'double_line_threshold': args.double_line_threshold,
                    'split_lines': args.split_lines,
                    'remove_side_margins': args.remove_side_margins
                }
                model_config_path = model_output_dir / 'config.json'
                if not model_config_path.is_file() or args.force_recompute:
                    model_config_path.write_text(json.dumps(model_config, indent=2))

        comm_img_paths = sorted(vs.get_comm_img_dir(comm_id).glob(f'{comm_id}*.png'), key=lambda x: x.stem)

        # Run the main loop
        for img_path in tqdm(comm_img_paths, desc=f'Processing {comm_id}'):
            page_image = cv2.imread(str(img_path))

            for model_name in args.models:
                output_path = vs.get_comm_olr_lines_dir(comm_id) / model_name / f'{img_path.stem}.json'

                if not args.force_recompute:

                    if output_path.is_file():
                        predictions = [geom.Shape(p) for p in json.loads(output_path.read_text())]

                    else:

                        if model_name in ['legacy', 'blla']:
                            predictions = all_models[model_name].predict(img_path=img_path)

                        elif model_name in ['legacy_adjusted', 'blla_adjusted']:
                            predictions = run_adjusted_model(model_name, img_path)

                        elif model_name == 'combined':
                            adjusted_model_predictions = {}
                            for adjusted_model in ['legacy_adjusted', 'blla_adjusted']:
                                adjusted_model_predictions_path = vs.get_comm_olr_lines_dir(comm_id) / adjusted_model / f'{img_path.stem}.json'

                                if adjusted_model_predictions_path.is_file():
                                    adjusted_model_predictions[adjusted_model] = [geom.Shape(p) for p in
                                                                                  json.loads(adjusted_model_predictions_path.read_text())]

                                else:
                                    adjusted_model_predictions[adjusted_model] = run_adjusted_model(adjusted_model, img_path)

                            predictions = all_models['combined'].predict(img_path=img_path,
                                                                         legacy_predictions=adjusted_model_predictions['legacy_adjusted'],
                                                                         blla_predictions=adjusted_model_predictions['blla_adjusted'])

                        else:
                            raise ValueError(f'Unknown segmentation model: {model_name}')


                else:
                    predictions = all_models[model_name].predict(img_path=img_path)

                if args.draw_dir is not None:
                    for p in predictions:
                        page_image = draw_box(p.bbox, page_image, stroke_color=colors[model_name],
                                              stroke_thickness=thicknesses[model_name])

                output_path.write_text(json.dumps([p.points for p in predictions]))

            if args.draw_dir is not None:
                cv2.imwrite(str(args.draw_dir / img_path.name), page_image)
