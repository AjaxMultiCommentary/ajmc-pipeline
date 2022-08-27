"""Use this script to bulk convert ocr runs to canonical commentaries"""
from ajmc.commons.image import draw_page_regions_lines_words
from ajmc.commons.variables import PATHS
from ajmc.text_processing.canonical_classes import CanonicalCommentary
from ajmc.text_processing.ocr_classes import OcrCommentary
import os

BASE_DATA_DIR = PATHS['base_dir']

commentaries = [
    ['Colonna1975', '18108i_kraken'],
    ['DeRomilly1976', '17g08V_kraken'],
    ['Ferrari1974', '17k0de_kraken'],
    ['Garvie1998', '17g0ao_kraken'],
    ['Kamerbeek1953', '17u09o_kraken'],
    ['Paduano1982', '17v0fZ_kraken'],
    ['Untersteiner1934', '17v0as_kraken'],
    ['Wecklein1894', '13p0am_lace_base'],
    ['bsb10234118', '13p07B_lace_retrained'],
    ['cu31924087948174', '15o0a0_lace_base_cu31924087948174-2021-05-23-21-05-58-porson-2021-05-23-14-27-27'],
    ['sophoclesplaysa05campgoog', '15o09Y_lace_base_sophoclesplaysa05campgoog-2021-05-23-21-38-49-porson-2021-05-23-14-27-27'],
    ['sophokle1v3soph', '13p0bP_lace_retrained']
]



# for comm_id, run_id in commentaries:
#     ocr_dir = os.path.join(BASE_DATA_DIR, comm_id, PATHS['ocr_runs'], run_id, 'outputs')
#     comm = OcrCommentary.from_ajmc_structure(ocr_dir=ocr_dir)
#     can = comm.to_canonical()
#     can.to_json(os.path.join(BASE_DATA_DIR, comm_id, PATHS['canonical'], run_id+'.json'))


#%%
comm = OcrCommentary.from_ajmc_structure('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/DeRomilly1976/ocr/runs/17g08V_kraken/outputs')
comm = comm.to_canonical()
comm.to_json('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/DeRomilly1976/canonical/v2/new.json')

#%% load the new and the old json canonical
import json
path = '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/DeRomilly1976/canonical/v2/17g08V_kraken.json'
with open(path,"r") as file:
    old = json.loads(file.read())

path='/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/DeRomilly1976/canonical/v2/new.json'
with open(path,"r") as file:
    new = json.loads(file.read())

#%% label the regions which have been added

# len(old['textcontainers']['page'])== len(new['textcontainers']['page'])
# j = 0
# added_regions = []
# for i in range(len(old['textcontainers']['region'])):
#     while old['textcontainers']['region'][i]['word_range'] != new['textcontainers']['region'][j]['word_range']:
#         new['textcontainers']['region'][j]['info']['region_type'] += '_added'
#         added_regions.append(new['textcontainers']['region'][j])
#         j+=1
#     j+=1
#
# path='/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/DeRomilly1976/canonical/v2/new_mod.json'
# with open(path, "w") as outfile:
#     json.dump(new, outfile, indent=4, ensure_ascii=False)


#%%
old = CanonicalCommentary.from_json('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/DeRomilly1976/canonical/v2/17g08V_kraken.json')
new = CanonicalCommentary.from_json('/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/DeRomilly1976/canonical/v2/28q0aU_tess.json')

for p_old, p_new in zip(old.children['page'], new.children['page']):
    try:
        draw_page_regions_lines_words(matrix=p_old.image.matrix.copy(),
                                      page=p_old,
                                      output_path=os.path.join('/Users/sven/Desktop/tests/', p_old.id + '_old.png'),
                                      )
        draw_page_regions_lines_words(matrix=p_new.image.matrix.copy(),
                                      page=p_new,
                                      output_path=os.path.join('/Users/sven/Desktop/tests/', p_new.id + '_new.png'), )

    except:
        continue


