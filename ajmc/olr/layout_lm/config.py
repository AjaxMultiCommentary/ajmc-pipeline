from typing import Optional
from ajmc.nlp.token_classification.config import initialize_config

rois = ['app_crit',
        'appendix',
        'bibliography',
        'commentary',
        'footnote',
        'index_siglorum',
        'introduction',
        'line_number_text',
        # 'line_number_commentary',
        'printed_marginalia',
        # 'handwritten_marginalia',
        'page_number',
        'preface',
        'primary_text',
        'running_header',
        'table_of_contents',
        'title',
        'translation',
        'other',
        # 'undefined'
        ]
regions_to_coarse_labels = {
    # Commentary
    'commentary': 'commentary',
    # Primary text
    'primary_text': 'primary_text',
    # Paratext
    'preface': 'paratext',
    'introduction': 'paratext',
    'footnote': 'paratext',
    'appendix': 'paratext',
    # Numbers
    'line_number_text': 'numbers',
    'line_number_commentary': 'numbers',
    'page_number': 'numbers',
    # App Crit
    'app_crit': 'app_crit',
    # Others
    'translation': 'others',
    'bibliography': 'others',
    'index_siglorum': 'others',
    'running_header': 'others',
    'table_of_contents': 'others',
    'title': 'others',
    'printed_marginalia': 'others',
    'handwritten_marginalia': 'others',
    'other': 'others',
    'undefined': 'others'
}
labels_to_ids = {
    # Commentary
    'commentary': 1,
    'primary_text': 2,
    'paratext': 3,
    'numbers': 4,
    'app_crit': 5,
    'others': 0
}
ids_to_labels = {v: k for k, v in labels_to_ids.items()}
ids_to_ner_labels = {'B-' + v: k for k, v in labels_to_ids.items()}

special_tokens = {
    'start': {'input_ids': 101, 'bbox': [0, 0, 0, 0], 'token_type_ids': 0, 'labels': -100, 'attention_mask': 1},
    'end': {'input_ids': 102, 'bbox': [1000, 1000, 1000, 1000], 'token_type_ids': 0, 'labels': -100,
            'attention_mask': 1},
    'pad': {'input_ids': 0, 'bbox': [0, 0, 0, 0], 'token_type_ids': 0, 'labels': -100, 'attention_mask': 0},
}


def create_olr_config(json_path: Optional[str] = None):
    config = initialize_config(json_path=json_path)
    config.regions_to_coarse_labels = regions_to_coarse_labels
    config.labels_to_ids = labels_to_ids
    config.ids_to_labels = ids_to_ner_labels
    config.model_inputs = ['input_ids', 'bbox', 'token_type_ids', 'attention_mask', 'image', 'labels']
    config.splits = ['train', 'dev']
    config.rois = rois
    config.special_tokens = special_tokens
    config.num_labels = len(list(labels_to_ids.keys()))

    return config
