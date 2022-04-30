from text_importation.classes import Commentary
from commons.variables import COMMENTARY_IDS

commentary_ids = [i for i in COMMENTARY_IDS if not i in ['Garvie1998', 'Kamerbeek1953']]
df_dict = {k: [] for k in ['commentary_id', 'pages', 'lines', 'words', 'paratext', 'primary_text', 'commentary']}

for commentary_id in commentary_ids:

    commentary = Commentary(commentary_id)
    comm_dict = {k: 0 for k in ['commentary_id', 'pages', 'lines', 'words', 'paratext', 'primary_text', 'commentary']}
    comm_dict['commentary_id'] = commentary_id


    for page in commentary.ocr_groundtruth_pages:
        comm_dict['pages'] += 1
        comm_dict['lines'] += len(page.lines)
        comm_dict['words'] += len(page.words)
        comm_dict['paratext'] += sum([len(r.words) for r in page.regions if r.region_type in ['introduction', 'preface', 'bibliography', 'footnote']])
        comm_dict['primary_text'] += sum([len(r.words) for r in page.regions if r.region_type == 'primary_text'])
        comm_dict['commentary'] += sum([len(r.words) for r in page.regions if r.region_type == 'commentary'])

    for k in df_dict.keys():
        df_dict[k].append(comm_dict[k])



import pandas as pd

df = pd.DataFrame(df_dict)
df.to_csv('groundtruth_stats_retrain.tsv', sep='\t', index = False)

