import json
import os
import random

from ajmc.commons.docstrings import docstrings, docstring_formatter
from transformers import LayoutLMv2TokenizerFast, LayoutLMv2ForTokenClassification, LayoutLMv2FeatureExtractor
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification, LayoutLMv3FeatureExtractor
from typing import List, Optional, Dict, Union, Tuple
from ajmc.nlp.token_classification.pipeline import train
from ajmc.commons.variables import COLORS, PATHS
from ajmc.nlp.token_classification.data_preparation.utils import align_from_tokenized, CustomDataset, \
    align_to_tokenized, align_labels
from ajmc.nlp.token_classification.model import predict_dataset
from ajmc.nlp.token_classification.pipeline import create_dirs
from ajmc.olr.utils import get_olr_split_page_ids
from ajmc.text_processing.canonical_classes import CanonicalCommentary
from PIL import Image

V3 = True

if V3:
    MODEL_INPUTS = ['input_ids',
                    'bbox',
                    'attention_mask',
                    'pixel_values']

else:
    MODEL_INPUTS = ['input_ids',
                    'bbox',
                    'token_type_ids',
                    'attention_mask',
                    'image']

ROBERTA_MODEL_INPUTS = ['input_ids', 'attention_mask']


# Functions
def normalize_bounding_rectangles(rectangle: List[List[int]], img_width: int, img_height: int, ):
    return [
        int(1000 * (rectangle[0][0] / img_width)),
        int(1000 * (rectangle[0][1] / img_height)),
        int(1000 * (rectangle[2][0] / img_width)),
        int(1000 * (rectangle[2][1] / img_height))
    ]


def get_data_dict_pages(data_dict: Dict[str, List[Dict[str, str]]],
                        sampling: Optional[Dict[str, float]] = None) -> Dict[str, List['CanonicalPage']]:
    """
    Args:
        data_dict: A dict of format `{'set': [{'id': 'comm_id_1', 'run':..., 'path': ...}, ...] }
        sampling: A dict of format `{'set': sample_size}` with 0>sample_size>1.
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

            page_ids = get_olr_split_page_ids(dict_['id'], dict_['split'])
            set_pages[set_] += [p for p in commentary.children['page'] if p.id in page_ids]

    if sampling:
        random.seed(42)
        for set_, sample_size in sampling.items():
            set_pages[set_] = random.sample(set_pages[set_], k=int(sample_size * len(set_pages[set_])))

    return set_pages


def page_to_layoutlmv2_encodings(page,
                                 rois,
                                 labels_to_ids,
                                 regions_to_coarse_labels,
                                 tokenizer,
                                 feature_extractor: 'FeatureExtractor',
                                 get_labels: bool = True,
                                 text_only: bool = False,
                                 unknownify_tokens: bool = False):
    # Get the lists of words, boxes and labels for a single page
    words = [w.text for r in page.children['region'] if r.info['region_type'] in rois for w in r.children['word']]

    if unknownify_tokens:
        words = [tokenizer.unk_token for _ in words]

    word_boxes = [normalize_bounding_rectangles(w.bbox.bbox, page.image.width, page.image.height)
                  for r in page.children['region'] if r.info['region_type'] in rois for w in r.children['word']]

    if not get_labels:
        word_labels = None
    else:
        word_labels = []
        for r in page.children['region']:
            if r.info['region_type'] in rois:
                for i, w in enumerate(r.children['word']):
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
            aligned_labels = [align_labels(e.word_ids, word_labels, labels_to_ids) for e in
                              encodings.encodings]

            encodings.data['labels'] = aligned_labels

        return encodings

    # Tokenize, truncate and pad
    encodings = tokenizer(text=words,
                          boxes=word_boxes,
                          word_labels=[labels_to_ids[l] for l in word_labels],
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
def prepare_data(page_sets: Dict[str, List['OcrPage']],
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
        page_sets: A dict containing a list of `OcrPage`s per split.
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
                split_encodings = page_to_layoutlmv2_encodings(page=page,
                                                               rois=rois,
                                                               labels_to_ids=labels_to_ids,
                                                               regions_to_coarse_labels=regions_to_coarse_labels,
                                                               tokenizer=tokenizer,
                                                               feature_extractor=feature_extractor,
                                                               unknownify_tokens=unknownify_tokens,
                                                               text_only=text_only)
            else:
                page_encodings = page_to_layoutlmv2_encodings(page=page,
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
def align_predicted_page(page: 'OcrPage',
                         rois,
                         labels_to_ids,
                         ids_to_labels,
                         regions_to_coarse_labels,
                         tokenizer,
                         feature_extractor,
                         model,
                         unknownify_tokens: bool = False,
                         text_only: bool = False,
                         ) -> Tuple[List['OcrWord'], List[str]]:
    encodings = page_to_layoutlmv2_encodings(page, rois=rois, labels_to_ids=labels_to_ids,
                                             regions_to_coarse_labels=regions_to_coarse_labels,
                                             tokenizer=tokenizer,
                                             feature_extractor=feature_extractor,
                                             get_labels=False,
                                             unknownify_tokens=unknownify_tokens,
                                             text_only=text_only)

    words = [w for r in page.children['region'] if r.info['region_type'] in rois for w in
             r.children['word']]  # this is the way words are selected in `page_to_layoutlmv2_encodings`

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
    from ajmc.olr.layout_lm.draw import draw_page_labels, draw_caption

    labels_to_colors = {l: c + tuple([127]) for l, c in
                        zip(labels_to_ids.keys(), list(COLORS['distinct'].values()) + list(COLORS['hues'].values()))}

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
    from ajmc.olr.layout_lm.config import create_olr_config

    config = create_olr_config(
        # json_path='/Users/sven/packages/ajmc/data/layoutlm/simple_config_local.json',
        json_path='/Users/sven/packages/ajmc/data/layoutlm/configs/1E_jebb_text_only.json',
        prefix=PATHS['base_dir']
    )
    main(config)
