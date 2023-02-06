import pandas as pd

from ajmc.commons import variables as vs
from ajmc.olr.utils import get_olr_splits_page_ids
from ajmc.text_processing.ocr_classes import OcrCommentary


MINIREF_PAGES = ['annalsoftacitusp00taci_0210',
                 'annalsoftacitusp00taci_0211',
                 'bsb10234118_0090',
                 'bsb10234118_0115',
                 'cu31924087948174_0063',
                 'cu31924087948174_0152',
                 'DeRomilly1976_0032',
                 'DeRomilly1976_0088',
                 'Ferrari1974_0050',
                 'Ferrari1974_0115',
                 'Garvie1998_0224',
                 'Garvie1998_0257',
                 'Kamerbeek1953_0098',
                 'Kamerbeek1953_0099',
                 'lestragdiesdeso00tourgoog_0113',
                 'lestragdiesdeso00tourgoog_0120',
                 'pvergiliusmaroa00virggoog_0199',
                 'pvergiliusmaroa00virggoog_0200',
                 'sophoclesplaysa05campgoog_0094',
                 'sophoclesplaysa05campgoog_0095',
                 'sophokle1v3soph_0047',
                 'sophokle1v3soph_0062',
                 'thukydides02thuc_0009',
                 'thukydides02thuc_0011',
                 'Untersteiner1934_0104',
                 'Untersteiner1934_0105',
                 'Wecklein1894_0016',
                 'Wecklein1894_0024',
                 'Colonna1975_0094',
                 'Colonna1975_0095']

dict_ = {x: [] for x in
         ['commentary_id', 'page_id', 'page_number', 'split', 'ocr_gt', 'olr_gt', 'loaded_inception', 'in_miniref',
          'annotate_linking', 'annotated', 'annotator', 'remarks']}

for comm_id in vs.ALL_COMM_IDS:
    comm = OcrCommentary.from_ajmc_data(id=comm_id)

    olr_page_ids = get_olr_splits_page_ids(comm_id)
    comm_section = comm.get_section('commentary')

    for page in comm_section.children.pages:
        if page.id in comm.ocr_gt_page_ids and page.id in olr_page_ids:
            dict_['commentary_id'].append(comm.id)
            dict_['page_id'].append(page.id)
            dict_['page_number'].append(page.id.split('_')[-1])
            dict_['split'].append('')
            dict_['ocr_gt'].append(True)
            dict_['olr_gt'].append(True)
            dict_['loaded_inception'].append(False)
            dict_['in_miniref'].append(page.id in MINIREF_PAGES)
            dict_['annotate_linking'].append(False)
            dict_['annotated'].append(False)
            dict_['annotator'].append('')
            dict_['remarks'].append('')

    for page in comm_section.children.pages:
        if dict_['commentary_id'].count(comm_id) >= 45:
            print(f'Already got 45 pages for {comm_id}')
            break
        if page.id in olr_page_ids and page.id not in dict_['page_id']:
            dict_['commentary_id'].append(comm.id)
            dict_['page_id'].append(page.id)
            dict_['page_number'].append(page.id.split('_')[-1])
            dict_['split'].append('')
            dict_['ocr_gt'].append(False)
            dict_['olr_gt'].append(True)
            dict_['loaded_inception'].append(False)
            dict_['in_miniref'].append(page.id in MINIREF_PAGES)
            dict_['annotate_linking'].append(False)
            dict_['annotated'].append(False)
            dict_['annotator'].append('')
            dict_['remarks'].append('')

    for page in comm_section.children.pages:
        if dict_['commentary_id'].count(comm_id) >= 45:
            print(f'Already got 45 pages for {comm_id}')
            break
        if page.id not in dict_['page_id']:
            dict_['commentary_id'].append(comm.id)
            dict_['page_id'].append(page.id)
            dict_['page_number'].append(page.id.split('_')[-1])
            dict_['split'].append('')
            dict_['ocr_gt'].append(False)
            dict_['olr_gt'].append(False)
            dict_['loaded_inception'].append(False)
            dict_['in_miniref'].append(page.id in MINIREF_PAGES)
            dict_['annotate_linking'].append(False)
            dict_['annotated'].append(False)
            dict_['annotator'].append('')
            dict_['remarks'].append('')

df = pd.DataFrame.from_dict(dict_)

df.to_csv('/Users/sven/Desktop/lemma_linkage_page_selection.tsv', sep='\t', index=False)

#%%
from pathlib import Path

for dir_ in Path('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/').iterdir():
    if dir_.is_dir():
        try:
            path = next(dir_.rglob('ner/annotation/xmi/*tess_base'))
        except StopIteration:
            print(dir_)
            continue

#%%
