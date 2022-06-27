import json
import os
import random

from ajmc.commons.docstrings import docstrings, docstring_formatter
from transformers import LayoutLMv2TokenizerFast, LayoutLMv2ForTokenClassification, LayoutLMv2FeatureExtractor
from typing import List, Optional, Dict, Union, Tuple
from ajmc.nlp.token_classification.pipeline import train
from ajmc.commons.variables import COLORS
from ajmc.nlp.token_classification.data_preparation.utils import align_from_tokenized, CustomDataset
from ajmc.nlp.token_classification.model import predict_dataset
from ajmc.nlp.token_classification.pipeline import create_dirs
from ajmc.text_importation.classes import Commentary
from PIL import Image
from ajmc.commons.miscellaneous import read_google_sheet
from ajmc.commons import variables


# Functions
def normalize_bounding_rectangles(rectangle: List[List[int]], img_width: int, img_height: int, ):
    return [
        int(1000 * (rectangle[0][0] / img_width)),
        int(1000 * (rectangle[0][1] / img_height)),
        int(1000 * (rectangle[2][0] / img_width)),
        int(1000 * (rectangle[2][1] / img_height))
    ]


def get_olr_split_pages(commentary: Commentary,
                        splits: List[str]) -> List['Page']:
    """Gets the data from splits on the olr_gt sheet."""

    olr_gt = read_google_sheet(variables.SPREADSHEETS_IDS['olr_gt'], 'olr_gt')

    filter_ = [(olr_gt['commentary_id'][i] == commentary.id and olr_gt['split'][i] in splits) for i in
               range(len(olr_gt['page_id']))]

    return [p for p in commentary.pages if p.id in list(olr_gt['page_id'][filter_])]



def get_data_dict_pages(data_dict: Dict[str, Dict[str, List[str]]],
                        sampling: Optional[Dict[str, float]] = None) -> Dict[str, List['Page']]:
    """
    Args:
        data_dict: A dict of format `{'set': {'ocr_dir':['split','split']}, }
        sampling: A dict of format `{'set': sample_size}` with 0>sample_size>1.
    """

    set_pages = {}
    for set_ in data_dict.keys():  # Iterate over set names, eg 'train', 'eval'
        set_pages[set_] = []
        for ocr_dir in data_dict[set_].keys():  # Iterate over ocr_dirs
            commentary = Commentary.from_ajmc_structure(ocr_dir=ocr_dir)
            set_pages[set_] += get_olr_split_pages(commentary, data_dict[set_][ocr_dir])

    if sampling:
        random.seed(42)
        for set_, sample_size in sampling.items():
            set_pages[set_] = random.sample(set_pages[set_], k=int(sample_size*len(set_pages[set_])))

    return set_pages


def page_to_layoutlmv2_encodings(page,
                                 rois,
                                 labels_to_ids,
                                 regions_to_coarse_labels,
                                 tokenizer,
                                 feature_extractor: Optional['FeatureExtractor'] = None,
                                 get_labels: bool = True,
                                 unknownify_tokens: bool = False):
    feature_extractor = feature_extractor if feature_extractor else \
        LayoutLMv2FeatureExtractor.from_pretrained('microsoft/layoutlmv2-base-uncased', apply_ocr=False)

    # Get the lists of words, boxes and labels for a single page
    words = [w.text for r in page.regions if r.region_type in rois for w in r.words]

    if unknownify_tokens:
        words = [tokenizer.unk_token for _ in words]

    word_boxes = [normalize_bounding_rectangles(w.coords.bounding_rectangle, page.image.width, page.image.height)
                  for r in page.regions if r.region_type in rois for w in r.words]

    word_labels = [labels_to_ids[regions_to_coarse_labels[r.region_type]]
                   for r in page.regions if r.region_type in rois for w in r.words] if get_labels else None

    # Tokenize, truncate and pad
    encodings = tokenizer(text=words,
                          boxes=word_boxes,
                          word_labels=word_labels,
                          truncation=True,
                          padding='max_length',
                          return_overflowing_tokens=True)

    # Add the image for each input
    image = Image.open(page.image.path).convert('RGB')
    image = feature_extractor(image)['pixel_values']
    encodings['image'] = [image[0].copy() for _ in range(len(encodings['input_ids']))]

    return encodings


@docstring_formatter(**docstrings)
def prepare_data(page_sets: Dict[str, List['Page']],
                 labels_to_ids: Dict[str, int],
                 regions_to_coarse_labels: Dict[str, str],
                 rois: List[str],
                 tokenizer,
                 unknownify_tokens:bool = False,
                 do_debug: bool = False
                 ) -> Dict[str, CustomDataset]:
    """Prepares data for LayoutLMV2.

    Args:
        page_sets: A dict containing a list of `Page`s per split.
        model_inputs_names: List of inputs the model wants (inputs_ids, attention_mask,...)
        labels_to_ids: {labels_to_ids}
        regions_to_coarse_labels:
        rois: The regions to focus on
        special_tokens: {special_tokens}
        tokenizer:
        unknownify_tokens:
        do_debug:

    """

    encodings = {}

    for s in page_sets.keys():
        split_encodings = None
        for i, page in enumerate(page_sets[s]):
            if split_encodings is None:
                split_encodings = page_to_layoutlmv2_encodings(page=page,
                                                               rois=rois,
                                                               labels_to_ids=labels_to_ids,
                                                               regions_to_coarse_labels=regions_to_coarse_labels,
                                                               tokenizer=tokenizer,
                                                               unknownify_tokens=unknownify_tokens)
            else:
                page_encodings = page_to_layoutlmv2_encodings(page=page,
                                                              rois=rois,
                                                              labels_to_ids=labels_to_ids,
                                                              regions_to_coarse_labels=regions_to_coarse_labels,
                                                              tokenizer=tokenizer,
                                                              unknownify_tokens=unknownify_tokens)
                for k in split_encodings.keys():
                    split_encodings[k] += page_encodings[k]

                split_encodings._encodings += page_encodings.encodings

            if i == 2 and do_debug:
                break

        encodings[s] = split_encodings

    # Todo : change this
    return {s: CustomDataset(encodings[s], ['input_ids', 'bbox', 'token_type_ids', 'attention_mask', 'image']) for s in page_sets.keys()}


# Todo : this must be a general function for token classification.
def align_predicted_page(page: 'Page',
                         rois,
                         labels_to_ids,
                         ids_to_labels,
                         regions_to_coarse_labels,
                         tokenizer,
                         model,
                         unknownify_tokens: bool = False
                         ) -> Tuple[List['Word'], List[str]]:

    encodings = page_to_layoutlmv2_encodings(page, rois=rois, labels_to_ids=labels_to_ids,
                                             regions_to_coarse_labels=regions_to_coarse_labels, tokenizer=tokenizer,
                                             get_labels=False, unknownify_tokens=unknownify_tokens)

    words = [w for r in page.regions if r.region_type in rois for w in
             r.words]  # this is the way words are selected in `page_to_layoutlmv2_encodings`

    dataset = CustomDataset(encodings=encodings,
                            model_inputs_names=['input_ids', 'bbox', 'token_type_ids', 'attention_mask', 'image'])

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
               model,
               output_dir: str,
               unknownify_tokens: bool = False,
               ):
    from ajmc.olr.layout_lm.draw import draw_page_labels, draw_caption

    labels_to_colors = {l: c + tuple([125]) for l, c in zip(labels_to_ids.keys(), COLORS['distinct'].values())}

    for page in pages:
        page_words, page_labels = align_predicted_page(page,
                                                       rois,
                                                       labels_to_ids,
                                                       ids_to_labels,
                                                       regions_to_coarse_labels,
                                                       tokenizer,
                                                       model,
                                                       unknownify_tokens=unknownify_tokens
                                                       )

        img = Image.open(page.image.path)
        img = draw_page_labels(img=img,
                         words=page_words,
                         labels=page_labels,
                         labels_to_colors=labels_to_colors)
        img = draw_caption(img, labels_to_colors=labels_to_colors)

        img.save(os.path.join(output_dir, page.id+'.png'))


def main(config):
    # config = create_olr_config('/Users/sven/packages/ajmc/data/configs/simple_config_local.json')
    create_dirs(config)

    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump(config.__dict__, f, skipkeys=True, indent=4, sort_keys=True,
                  default=lambda o: '<not serializable>')

    tokenizer = LayoutLMv2TokenizerFast.from_pretrained(config.model_name_or_path)
    model = LayoutLMv2ForTokenClassification.from_pretrained(config.model_name_or_path, num_labels=config.num_labels)

    pages = get_data_dict_pages(data_dict=config.data_dirs_and_sets, sampling=config.sampling)

    datasets = prepare_data(page_sets=pages,
                            labels_to_ids=config.labels_to_ids,
                            regions_to_coarse_labels=config.regions_to_coarse_labels,
                            rois=config.rois,
                            tokenizer=tokenizer,
                            unknownify_tokens=config.unknownify_tokens,
                            do_debug=config.do_debug)

    if config.do_train:
        train(config=config, model=model, train_dataset=datasets['train'], eval_dataset=datasets['eval'],
              tokenizer=tokenizer)

    # draw
    if config.do_draw:
        draw_pages(pages=pages['eval'],
                   rois=config.rois,
                   labels_to_ids=config.labels_to_ids,
                   ids_to_labels=config.ids_to_raw_labels,
                   regions_to_coarse_labels=config.regions_to_coarse_labels,
                   tokenizer=tokenizer,
                   model=model,
                   output_dir=config.predictions_dir,
                   unknownify_tokens=config.unknownify_tokens)

