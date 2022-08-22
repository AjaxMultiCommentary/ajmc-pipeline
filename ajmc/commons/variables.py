import re
import logging

PATHS = {
    # 'base_dir': '/mnt/ajmcdata1/drive_cached/AjaxMultiCommentary/data/commentaries/commentaries_data/',
    'base_dir': '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data',
    'drive_base_dir': '/content/drive/MyDrive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/',
    'cluster_base_dir': '/mnt/ajmcdata1/drive_cached/AjaxMultiCommentary/data/commentaries/commentaries_data',
    'schema': 'data/templates/page.schema.json',
    'groundtruth': 'ocr/groundtruth/evaluation',
    'png': 'images/png',
    'via_path': 'olr/via_project.json',
    'json': 'canonical',
    'xmi': 'ner/annotation',
    'typesystem': 'data/TypeSystem.xml',
    'olr_initiation': 'olr/annotation/project_initiation',
    'ocr': 'ocr/runs',
    'canonical': 'canonical/v2'
}

# todo 👁️
def get_path():
    pass


FOLDER_STRUCTURE_PATHS = {
    # Only relative paths
    # Placeholder pattern are between []
    'ocr_outputs_dir': '[commentary_id]/ocr/runs/[ocr_run]/outputs'
}

# Sheet names corresponds to the dictionary's keys
SPREADSHEETS_IDS = {
    'metadata': '1jaSSOF8BWij0seAAgNeGe3Gtofvg9nIp_vPaSj5FtjE',
    'olr_gt': '1_hDP_bGDNuqTPreinGS9-ShnXuXCjDaEbz-qEMUSito'
}

VIA_CSV_DICT_TEMPLATE = {'filename': [],
                         'file_size': [],
                         'file_attributes': [],
                         'region_count': [],
                         'region_id': [],
                         'region_shape_attributes': [],
                         'region_attributes': []}

COMMENTARY_IDS = ['Colonna1975', 'DeRomilly1976', 'Ferrari1974', 'Finglass2011', 'Garvie1998', 'Kamerbeek1953',
                  'Paduano1982', 'Untersteiner1934', 'Wecklein1894', 'bsb10234118', 'cu31924087948174',
                  'lestragdiesdeso00tourgoog', 'sophoclesplaysa05campgoog', 'sophokle1v3soph']

PD_COMMENTARY_IDS = ['bsb10234118', 'cu31924087948174', 'sophoclesplaysa05campgoog', 'sophokle1v3soph', 'Wecklein1894']

ORDERED_OLR_REGION_TYPES = ['commentary',
                            'primary_text',
                            'preface',
                            'translation',
                            'introduction',
                            'line_number_text',
                            'line_number_commentary',
                            'page_number',
                            'appendix',
                            'app_crit',
                            'bibliography',
                            'footnote',
                            'index_siglorum',
                            'running_header',
                            'table_of_contents',
                            'title',
                            'printed_marginalia',
                            'handwritten_marginalia',
                            'other',
                            'undefined',
                            'line_region',  # added only to make sure every word has a region
                            ]

EXCLUDED_REGION_TYPES = ['line_number_commentary', 'handwritten_marginalia', 'undefined', 'line_region']
ROIS = [rt for rt in ORDERED_OLR_REGION_TYPES if rt not in EXCLUDED_REGION_TYPES]

REGION_TYPES_TO_COARSE_LABELS = {
    # Commentary
    'commentary': 'commentary',
    # Primary text
    'primary_text': 'primary_text',
    # Footnotes
    'footnote': 'footnote',
    # Running header
    'running_header': 'running_header',
    # Paratext
    'preface': 'paratext',
    'introduction': 'paratext',
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
    'table_of_contents': 'others',
    'title': 'others',
    'printed_marginalia': 'others',
    'handwritten_marginalia': 'others',
    'other': 'others',
    'undefined': 'others',
    'line_region': 'others'
}

REGION_TYPES_TO_FINE_LABELS = {k:k for k in REGION_TYPES_TO_COARSE_LABELS.keys()}

MINIREF_PAGES = [
    'cu31924087948174_0035',
    'cu31924087948174_0063',
    'sophoclesplaysa05campgoog_0014',
    'sophoclesplaysa05campgoog_0146',
    'sophoclesplaysa05campgoog_0288',
    'Wecklein1894_0007',
    'Wecklein1894_0016',
    'Wecklein1894_0080',
    'Wecklein1894_0087',
    'sophokle1v3soph_0017',
    'sophokle1v3soph_0049',
    'sophokle1v3soph_0085',
    'sophokle1v3soph_0125',
]

CHARSETS = {
    'latin': re.compile(r'([A-Za-z]|[\u00C0-\u00FF]|\u0152|\u0153)', re.UNICODE),
    'greek': re.compile(r'([\u0373-\u03FF]|[\u1F00-\u1FFF]|\u0300|\u0301|\u0313|\u0314|\u0345|\u0342|\u0308)',
                        re.UNICODE),
    'numbers': re.compile(r'([0-9])', re.UNICODE),
    'punctuation': re.compile(r'([\u0020-\u002F]|[\u003A-\u003F]|[\u005B-\u0060]|[\u007B-\u007E]|\u00A8|\u00B7)',
                              re.UNICODE)
}

COLORS = {
    # https://coolors.co/b2001e-f02282-461e44-3b9ff1-37507d-125b4f-98e587-ffc802-af7159-b5a267
    'distinct': {
        'red': (178, 0, 30),
        'pink': (240, 34, 130),
        'blue': (59, 159, 241),
        'green': (152, 229, 135),
        'yellow': (255, 200, 2),
        'brown': (175, 113, 89),
        'dark_green': (18, 91, 79),
        'purple': (70, 30, 68),
        'dark_blue': (55, 80, 125),
        'ecru': (181, 162, 103),
    },
    # https://coolors.co/f72585-b5179e-7209b7-560bad-480ca8-3a0ca3-3f37c9-4361ee-4895ef-4cc9f0
    'hues': {
        'pink': (247, 37, 133),
        'byzantine': (181, 23, 158),
        'purple1': (114, 9, 183),
        'purple2': (86, 11, 173),
        'trypan_blue1': (72, 12, 168),
        'trypan_blue2': (58, 12, 163),
        'persian_blue': (63, 55, 201),
        'ultramarine_blue': (67, 97, 238),
        'dodger_blue': (72, 149, 239),
        'vivid_sky_blue': (76, 201, 240)
    }
    # Other color palettes
    # https://coolors.co/97dffc-93caf6-8eb5f0-858ae3-7364d2-613dc1-5829a7-4e148c-461177-3d0e61
    # https://coolors.co/b7094c-a01a58-892b64-723c70-5c4d7d-455e89-2e6f95-1780a1-0091ad
    # https://coolors.co/0081a7-00afb9-fdfcdc-fed9b7-f07167
}


TEXTCONTAINER_TYPES = ['commentary', 'page', 'region', 'line', 'word']

LOGGING_LEVEL = logging.ERROR