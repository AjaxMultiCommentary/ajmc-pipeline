import os
import json
from seqeval.metrics import performance_measure
from transformers import LayoutLMv2TokenizerFast, LayoutLMv2ForTokenClassification

from ajmc.commons.variables import COLORS
from ajmc.nlp.token_classification.evaluation import evaluate_dataset
from ajmc.nlp.token_classification.model import predict_dataset
from ajmc.olr.layout_lm.config import labels_to_ids, rois, regions_to_coarse_labels, ids_to_labels
from ajmc.olr.layout_lm.layoutlm import get_olr_split_pages, draw_pages, prepare_data, get_data_dict_pages
from ajmc.text_processing.ocr_classes import OcrCommentary

base_path = '/Users/sven/drive/layout_lm_tests'
# %%

# for fname in next(os.walk(base_path))[1]:  # Walk in dirs only
fname = next(os.walk(base_path))[1][0]
if not fname.startswith('z'):
    with open(os.path.join(base_path, fname, 'config.json'), "r") as file:
        config = json.loads(file.read())

    model_name_or_path = os.path.join(base_path, fname, 'model')

    pages = []
    old_prefix = '/content/drive/MyDrive/'
    new_prefix = '/Users/sven/drive/'

    config['data_dirs_and_sets'] = {'eval': {
        os.path.join(new_prefix, ocr_dir[len(old_prefix):]): splits
        for ocr_dir, splits in config['data_dirs_and_sets']['eval'].items()
    }
    }

    tokenizer = LayoutLMv2TokenizerFast.from_pretrained(model_name_or_path)
    model = LayoutLMv2ForTokenClassification.from_pretrained(model_name_or_path)

    pages = get_data_dict_pages(data_dict=config['data_dirs_and_sets'], sampling=config['sampling'])
    datasets = prepare_data(pages, labels_to_ids=labels_to_ids, regions_to_coarse_labels=regions_to_coarse_labels,
                            rois=rois,
                            tokenizer=tokenizer, unknownify_tokens=config['unknownify_tokens'])

    predict_dataset(dataset=datasets['eval'], model=model)

#%%
    evaluation_results = evaluate_dataset(dataset=datasets['eval'], model=model, batch_size=1, device=None, ids_to_labels=ids_to_labels)


#%%
