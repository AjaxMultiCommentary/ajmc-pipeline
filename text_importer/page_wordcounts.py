import os
from commons.variables import PATHS, open_commentaries
from text_importer.pagexml import PagexmlPage
import pandas as pd

sheet = {c: {'page': [], 'region': [], 'word_count': []} for c in open_commentaries}

for commentary_id in open_commentaries:
    print("Process commentary  " + commentary_id)
    pages_ids = [p[:-4] for p in os.listdir(os.path.join(PATHS['base_dir'], commentary_id, PATHS['png'])) if
                 p.endswith('.png')]
    page_ids = sorted(pages_ids)

    for page_id in pages_ids:
        if int(page_id.split('_')[-1]) % 20 == 0:
            print("processing page  " + page_id)
        page = PagexmlPage(page_id)
        regions = page.get_olrregions(['introduction', 'preface', 'footnotes', 'commentary'])
        region_counts = {r.region_type: 0 for r in regions}

        for region in regions:
            region_counts[region.region_type] += len(region._get_children("word"))

        for region, count in region_counts.items():
            sheet[commentary_id]['page'].append(page.id.split('_')[1])
            sheet[commentary_id]['region'].append(region)
            sheet[commentary_id]['word_count'].append(count)

with pd.ExcelWriter('/Users/sven/Desktop/temp.xlsx') as writer:
    for commentary_id in open_commentaries:
        df = pd.DataFrame(sheet[commentary_id])
        df.to_excel(writer, sheet_name=commentary_id, index=False)
