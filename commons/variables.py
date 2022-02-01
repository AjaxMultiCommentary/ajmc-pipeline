PATHS = {'base_dir': '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/',
         'schema': '/Users/sven/ajmc/commons/page.schema.json',
         'png': 'images/png',
         'via_project': 'olr/via_project.json',
         'pagexml': 'ocr/ocrs/tess_final/',  # todo change this system to custom folders !!! (e.g. you can chose the folder in args
         'tesshocr': 'ocr/ocrs/tess_hocr',
         # 'krakenhocr': 'ocr/evaluation/groundtruth/html',
         'krakenhocr': 'ocr/ocrs/lace_retrained_sophoclesplaysa05campgoog-2021-05-24-08-15-12-porson-with-sophoclesplaysa05campgoog-2021-05-23-22-17-38/outputs',
         'json': 'canonical',
         'xmi': 'ner/annotation',
         'typesystem': '/Users/sven/ajmc/data/xmi_annotation/TypeSystem.xml'
         }

METADATA_SPREADSHEET_ID = '1jaSSOF8BWij0seAAgNeGe3Gtofvg9nIp_vPaSj5FtjE'
METADATA_WORKSHEET_NAME = 'metadata'

via_csv_dict_template = {'filename': [],
                         'file_size': [],
                         'file_attributes': [],
                         'region_count': [],
                         'region_id': [],
                         'region_shape_attributes': [],
                         'region_attributes': []}

pd_commentaries = ['bsb10234118', 'cu31924087948174', 'sophoclesplaysa05campgoog', 'sophokle1v3soph', 'Wecklein1894']

olr_regions_types = ['app_crit',
                     'appendix',
                     'bibliography',
                     'commentary',
                     'footnote',
                     'index_siglorum',
                     'introduction',
                     'line_number_text',
                     'line_number_commentary',
                     'printed_marginalia',
                     'handwritten_marginalia',
                     'page_number',
                     'preface',
                     'primary_text',
                     'running_header',
                     'table_of_contents',
                     'title',
                     'translation',
                     'other',
                     'undefined'
                     ]

miniref_pages = [
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
