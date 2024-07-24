import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from PIL import Image
from transformers import TrainingArguments

from ajmc.commons import variables as vs
from ajmc.commons.docstrings import docstring_formatter, docstrings
from ajmc.commons.miscellaneous import get_ajmc_logger
from ajmc.nlp.token_classification.data_preparation.utils import align_from_tokenized, align_labels_to_tokenized, \
    CustomDataset
from ajmc.nlp.token_classification.model import predict_dataset
from ajmc.nlp.token_classification.pipeline import create_dirs, train
from ajmc.olr.utils import get_olr_splits_page_ids
from ajmc.text_processing.canonical_classes import CanonicalCommentary

logger = get_ajmc_logger(__name__)
V3 = True


def create_default_config() -> Dict[str, Any]:
    """Creates a default token-classification config."""

    config = dict()

    # ================ PATHS AND DIRS ==================================================================================
    config['train_path']: str = None  # Absolute path to the tsv data file to train on # Required: False
    config['train_url']: str = None  # url to the tsv data file to train on # Required: False
    config['eval_path']: str = None  # Absolute path to the tsv data file to evaluate on # Required: False
    config['eval_url']: str = None  # url to the tsv data file to evaluate on # Required: False
    config[
        'output_dir']: str = None  # Absolute path to the directory in which outputs are to be stored # Required: False
    config[
        'hipe_script_path']: str = None  # The path the CLEF-HIPE-evaluation script. This parameter is required if ``do_hipe_eval`` is True # Required: False
    config[
        'config_path']: str = None  # The path to a config json file from which to extract config. Overwrites other specified config # Required: False
    config['predict_paths']: list = []  # A list of tsv files to predict # Required: False
    config['predict_urls']: list = []  # A list of tsv files-urls to predict # Required: False

    # ================ DATA RELATED ====================================================================================
    config['labels_column']: str = None  # Name of the tsv col to extract labels from # Required: False
    config['unknownify_tokens']: bool = False  # Sets all tokens to '[UNK]'. Useful for ablation experiments. # Required: False
    # config['sampling'] # 👁️ add ?

    # ================ MODEL INFO ======================================================================================
    config[
        'model_name_or_path']: str = None  # Absolute path to model directory  or HF model name (e.g. 'bert-base-cased') # Required: False

    # =================== ACTIONS ======================================================================================
    config['do_train']: bool = False  # whether to train. Leave to false if you just want to evaluate
    config['do_eval']: bool = False  # Performs CLEF-HIPE evaluation, alone or at the end of training if ``do_train``.
    config['do_predict']: bool = False  # Predicts on ``predict_urls`` or/and ``predict_paths``
    config['evaluate_during_training']: bool = False  # Whether to evaluate during training.
    config['do_debug']: bool = False  # Breaks all loops after a single iteration for debugging
    config['overwrite_output_dir']: bool = False  # Whether to overwrite the output dir
    config[
        'do_early_stopping']: bool = False  # Breaks stops training after ``early_stopping_patience`` epochs without improvement.

    # =============================== TRAINING PARAMETERS ==============================================================
    config['device_name']: str = "cuda:0"  # Device in the format 'cuda:1', 'cpu'
    config['epochs']: int = 3  # Total number of training epochs to perform.
    config['early_stopping_patience']: int = 3  # Number of epochs to wait for early stopping
    config['seed']: int = 42  # Random seed
    config['batch_size']: int = 8  # Batch size per device.
    config['gradient_accumulation_steps']: int = 1  # Number of steps to accumulate before performing backpropagation.

    # ===================================== HF PARAMETERS ==============================================================
    default_hf_args: dict = vars(TrainingArguments(''))
    config.update(**{arg: default_hf_args[arg] for arg in ['local_rank', 'weight_decay', 'max_grad_norm',
                                                           'adam_epsilon', 'learning_rate', 'warmup_steps']})
    return config


def parse_config_from_json(json_path: Path) -> Dict[str, Any]:
    """Parses config from a json file.

    Also transforms ``config['device_name']`` to ``torch.device(config['device_name'])``, raising an error if ``cuda[:#]``
    is set but not available.
    """

    json_args = json.loads(json_path.read_text(encoding='utf-8'))

    config = create_default_config()
    config.update(**{arg: json_args[arg] for arg in json_args.keys()})

    if config['device_name'].startswith("cuda") and not torch.cuda.is_available():
        logger.error("You set ``device_name`` to {} but cuda is not available, setting device to cpu.".format(config['device_name']))
        config['device'] = torch.device('cpu')

    else:
        config['device'] = torch.device(config['device_name'])

    return config

if V3:
    from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification, LayoutLMv3FeatureExtractor

    MODEL_INPUTS = ['input_ids',
                    'bbox',
                    'attention_mask',
                    'pixel_values']

else:
    from transformers import LayoutLMv2TokenizerFast, LayoutLMv2ForTokenClassification, LayoutLMv2FeatureExtractor

    MODEL_INPUTS = ['input_ids',
                    'bbox',
                    'token_type_ids',
                    'attention_mask',
                    'image']

ROBERTA_MODEL_INPUTS = ['input_ids', 'attention_mask']


def create_olr_config(json_path: Path,
                      prefix: str):
    config = parse_config_from_json(json_path=str(json_path))

    for set_, data_list in config['data'].items():
        for dict_ in data_list:
            dict_['path'] = os.path.join(prefix, dict_['id'], vs.PATHS['canonical'], dict_['run'] + '.json')

    config['rois'] = [rt for rt in vs.ORDERED_OLR_REGION_TYPES if rt not in config['excluded_region_types']]
    region_types_to_labels = {k: l for k, l in config['region_types_to_labels'].items() if k in config['rois']}
    config['labels_to_ids'] = {l: i for i, l in enumerate(sorted(set(region_types_to_labels.values())))}
    config['ids_to_labels'] = {l: i for i, l in config['labels_to_ids'].items()}
    config['num_labels'] = len(list(config['labels_to_ids'].keys()))
    if 'sampling' not in config.keys():
        config['sampling'] = None

    return config


# Functions
def normalize_bounding_box(bbox: vs.BoxType, img_width: int, img_height: int, ):
    return [
        int(1000 * (bbox[0][0] / img_width)),
        int(1000 * (bbox[0][1] / img_height)),
        int(1000 * (bbox[1][0] / img_width)),
        int(1000 * (bbox[1][1] / img_height))
    ]


def get_data_dict_pages(data_dict: Dict[str, List[Dict[str, str]]],
                        sampling: Optional[Dict[str, float]] = None) -> Dict[str, List['CanonicalPage']]:
    """
    Args:
        data_dict: A dict of format ``{'set': [{'id': 'comm_id_1', 'run':..., 'path': ...}, ...] }``
        sampling: A dict of format ``{'set': sample_size}`` with 0>sample_size>1.
    """

    set_pages = {}
    commentaries = {}

    for set_, dicts in data_dict.items():  # Iterate over set names, eg 'train', 'eval'
        set_pages[set_] = []
        for dict_ in dicts:
            try:
                commentary = commentaries[dict_['id']]
            except KeyError:
                commentary = CanonicalCommentary.from_json(json_path=dict_['path'])
                commentaries[dict_['id']] = commentary

            page_ids = get_olr_splits_page_ids(dict_['id'], [dict_['split']])
            set_pages[set_] += [p for p in commentary.children.pages if p.id in page_ids]

    if sampling:
        random.seed(42)
        for set_, sample_size in sampling.items():
            set_pages[set_] = random.sample(set_pages[set_], k=int(sample_size * len(set_pages[set_])))

    return set_pages


def page_to_layoutlm_encodings(page,
                               rois,
                               labels_to_ids,
                               regions_to_coarse_labels,
                               tokenizer,
                               feature_extractor: 'FeatureExtractor',
                               get_labels: bool = True,
                               text_only: bool = False,
                               unknownify_tokens: bool = False):
    # Get the lists of words, boxes and labels for a single page
    words = [w.text for r in page.children.regions if r.info['region_type'] in rois for w in r.children.words]

    if unknownify_tokens:
        words = [tokenizer.unk_token for _ in words]

    word_boxes = [normalize_bounding_box(w.bbox.bbox, page.image.width, page.image.height)
                  for r in page.children.regions if r.info['region_type'] in rois for w in r.children.words]

    if not get_labels:
        word_labels = None
    else:
        word_labels = []
        for r in page.children.regions:
            if r.info['region_type'] in rois:
                for i, w in enumerate(r.children.words):
                    word_labels.append(regions_to_coarse_labels[r.info['region_type']])
                    # if i != 0:
                    #     word_labels.append(labels_to_ids['O'])
                    #     # word_labels.append(labels_to_ids['I-'+ region_types_to_labels[r.info['region_type']]])
                    # else:
                    #     word_labels.append(labels_to_ids[region_types_to_labels[r.info['region_type']]])

    if text_only:
        encodings = tokenizer(text=words,
                              is_split_into_words=True,
                              truncation=True,
                              padding='max_length',
                              return_overflowing_tokens=True)

        if get_labels:
            aligned_labels = [align_labels_to_tokenized(e.word_ids, word_labels, labels_to_ids) for e in
                              encodings.encodings]

            encodings.data['labels'] = aligned_labels

        return encodings

    # Tokenize, truncate and pad
    encodings = tokenizer(text=words,
                          boxes=word_boxes,
                          word_labels=[labels_to_ids[l] for l in word_labels] if get_labels else None,
                          truncation=True,
                          padding='max_length',
                          return_overflowing_tokens=True)

    # Add the image for each input
    image = Image.open(page.image.path).convert('RGB')
    image = feature_extractor(image)['pixel_values']

    if V3:
        encodings['pixel_values'] = [image[0].copy() for _ in range(len(encodings['input_ids']))]
    else:
        encodings['image'] = [image[0].copy() for _ in range(len(encodings['input_ids']))]

    return encodings


@docstring_formatter(**docstrings)
def prepare_data(page_sets: Dict[str, List['RawPage']],
                 labels_to_ids: Dict[str, int],
                 regions_to_coarse_labels: Dict[str, str],
                 rois: List[str],
                 tokenizer,
                 feature_extractor,
                 unknownify_tokens: bool = False,
                 text_only: bool = False,
                 do_debug: bool = False
                 ) -> Dict[str, CustomDataset]:
    """Prepares data for LayoutLMV2.

    Args:
        page_sets: A dict containing a list of ``RawPage``\ s per split.
        labels_to_ids: {labels_to_ids}
        regions_to_coarse_labels:
        rois: The regions to focus on
        tokenizer:
        unknownify_tokens:
        do_debug:

    """

    encodings_split_dict = {}

    for s in page_sets.keys():
        split_encodings = None
        for i, page in enumerate(page_sets[s]):
            if split_encodings is None:
                split_encodings = page_to_layoutlm_encodings(page=page,
                                                             rois=rois,
                                                             labels_to_ids=labels_to_ids,
                                                             regions_to_coarse_labels=regions_to_coarse_labels,
                                                             tokenizer=tokenizer,
                                                             feature_extractor=feature_extractor,
                                                             unknownify_tokens=unknownify_tokens,
                                                             text_only=text_only)
            else:
                page_encodings = page_to_layoutlm_encodings(page=page,
                                                            rois=rois,
                                                            labels_to_ids=labels_to_ids,
                                                            regions_to_coarse_labels=regions_to_coarse_labels,
                                                            tokenizer=tokenizer,
                                                            feature_extractor=feature_extractor,
                                                            unknownify_tokens=unknownify_tokens,
                                                            text_only=text_only)
                for k in split_encodings.keys():
                    split_encodings[k] += page_encodings[k]

                split_encodings._encodings += page_encodings.encodings

            if i == 2 and do_debug:
                break

        encodings_split_dict[s] = split_encodings

    if text_only:
        return {s: CustomDataset(encodings=encodings_split_dict[s],
                                 model_inputs_names=ROBERTA_MODEL_INPUTS)
                for s in page_sets.keys()}
    else:
        return {s: CustomDataset(encodings_split_dict[s], MODEL_INPUTS) for s in
                page_sets.keys()}


# todo 👁️ this must be a general function for token classification.
def align_predicted_page(page: 'Page',
                         rois,
                         labels_to_ids,
                         ids_to_labels,
                         regions_to_coarse_labels,
                         tokenizer,
                         feature_extractor,
                         model,
                         unknownify_tokens: bool = False,
                         text_only: bool = False,
                         ) -> Tuple[List['Word'], List[str]]:
    encodings = page_to_layoutlm_encodings(page, rois=rois, labels_to_ids=labels_to_ids,
                                           regions_to_coarse_labels=regions_to_coarse_labels,
                                           tokenizer=tokenizer,
                                           feature_extractor=feature_extractor,
                                           get_labels=False,
                                           unknownify_tokens=unknownify_tokens,
                                           text_only=text_only)

    words = [w for r in page.children.regions if r.info['region_type'] in rois for w in
             r.children.words]  # this is the way words are selected in ``page_to_layoutlm_encodings``

    if text_only:
        dataset = CustomDataset(encodings=encodings, model_inputs_names=ROBERTA_MODEL_INPUTS)
    else:
        dataset = CustomDataset(encodings=encodings,
                                model_inputs_names=MODEL_INPUTS)

    # Merge tokens offsets together
    word_ids_list: List[Union[int, None]] = [el for encoding in dataset.encodings.encodings for el in encoding.word_ids]

    # Predictions is an array, with the same shape as dataset.encodings
    predictions = predict_dataset(dataset=dataset, model=model, batch_size=1)
    # Merge predictions together
    prediction_list = predictions.tolist()  # A list of lists
    prediction_list = [el for sublist in prediction_list for el in sublist]
    aligned_labels = align_from_tokenized(word_ids_list, prediction_list)
    aligned_labels = [ids_to_labels[l] for l in aligned_labels]

    assert len(aligned_labels) == len(words)

    return words, aligned_labels


def draw_pages(pages,
               rois,
               labels_to_ids,
               ids_to_labels,
               regions_to_coarse_labels,
               tokenizer,
               feature_extractor,
               model,
               output_dir: str,
               unknownify_tokens: bool = False,
               text_only: bool = False,
               ):
    from ajmc.olr.layoutlm.draw import draw_page_labels, draw_caption

    labels_to_colors = {l: c + tuple([127]) for l, c in
                        zip(labels_to_ids.keys(), list(vs.COLORS['distinct'].values()) + list(vs.COLORS['hues'].values()))}

    for page in pages:
        page_words, page_labels = align_predicted_page(page,
                                                       rois,
                                                       labels_to_ids,
                                                       ids_to_labels,
                                                       regions_to_coarse_labels,
                                                       tokenizer,
                                                       feature_extractor,
                                                       model,
                                                       unknownify_tokens=unknownify_tokens,
                                                       text_only=text_only
                                                       )

        img = Image.open(page.image.path)
        img = draw_page_labels(img=img,
                               words=page_words,
                               labels=page_labels,
                               labels_to_colors=labels_to_colors)
        img = draw_caption(img, labels_to_colors=labels_to_colors)

        img.save(os.path.join(output_dir, page.id + '.png'))


def main(config):
    # config = create_olr_config('/Users/sven/packages/ajmc/data/configs/simple_config_local.json')
    create_dirs(config)

    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, skipkeys=True, indent=4, sort_keys=True,
                  default=lambda o: '<not serializable>')

    if config['text_only']:
        from transformers import RobertaTokenizerFast, RobertaForTokenClassification
        tokenizer = RobertaTokenizerFast.from_pretrained(config['model_name_or_path'], add_prefix_space=True)
        model = RobertaForTokenClassification.from_pretrained(config['model_name_or_path'],
                                                              num_labels=config['num_labels'])
        feature_extractor = None
    else:
        if V3:
            tokenizer = LayoutLMv3TokenizerFast.from_pretrained(config['model_name_or_path'])
            model = LayoutLMv3ForTokenClassification.from_pretrained(config['model_name_or_path'],
                                                                     num_labels=config['num_labels'])
            feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(config['model_name_or_path'],
                                                                           apply_ocr=False)
        else:
            tokenizer = LayoutLMv2TokenizerFast.from_pretrained(config['model_name_or_path'])
            model = LayoutLMv2ForTokenClassification.from_pretrained(config['model_name_or_path'],
                                                                     num_labels=config['num_labels'])
            feature_extractor = LayoutLMv2FeatureExtractor.from_pretrained(config['model_name_or_path'],
                                                                           apply_ocr=False)

    pages = get_data_dict_pages(data_dict=config['data'], sampling=config['sampling'])

    datasets = prepare_data(page_sets=pages,
                            labels_to_ids=config['labels_to_ids'],
                            regions_to_coarse_labels=config['region_types_to_labels'],
                            rois=config['rois'],
                            tokenizer=tokenizer,
                            feature_extractor=feature_extractor,
                            unknownify_tokens=config['unknownify_tokens'],
                            text_only=config['text_only'],
                            do_debug=config['do_debug'])

    if config['do_train']:
        train(config=config, model=model, train_dataset=datasets['train'], eval_dataset=datasets['eval'],
              tokenizer=tokenizer)

    # draw
    if config['do_draw']:
        draw_pages(pages=pages['eval'],
                   rois=config['rois'],
                   labels_to_ids=config['labels_to_ids'],
                   ids_to_labels=config['ids_to_labels'],
                   regions_to_coarse_labels=config['region_types_to_labels'],
                   tokenizer=tokenizer,
                   feature_extractor=feature_extractor,
                   model=model,
                   output_dir=config['predictions_dir'],
                   unknownify_tokens=config['unknownify_tokens'],
                   text_only=config['text_only'])


if __name__ == '__main__':
    config = create_olr_config(
            # json_path='/Users/sven/packages/ajmc/data/layoutlm/simple_config_local.json',
            json_path='/Users/sven/packages/ajmc/data/layoutlm/configs/1E_jebb_text_only.json',
            prefix=vs.COMMS_DATA_DIR
    )
    main(config)
