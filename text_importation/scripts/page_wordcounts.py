from common_utils.variables import pd_commentaries
from text_importation.classes import Commentary

import pandas as pd

sheet = {c: {'page': [], 'region': [], 'word_count': []} for c in pd_commentaries}

for commentary_id in ['sophoclesplaysa05campgoog']:
    print("Process commentary  " + commentary_id)
    commentary = Commentary(commentary_id, )

    for page in commentary.pages:
        if int(page.id.split('_')[-1]) % 20 == 0:
            print("processing page  " + page.id)
        regions = [r for r in page.regions if r.region_type in ['introduction', 'preface', 'footnotes', 'commentary']]
        region_counts = {r.region_type: 0 for r in regions}

        for region in regions:
            region_counts[region.region_type] += len(region.words)

        for region, count in region_counts.items():
            sheet[commentary_id]['page'].append(page.id.split('_')[1])
            sheet[commentary_id]['region'].append(region)
            sheet[commentary_id]['word_count'].append(count)

with pd.ExcelWriter('/Users/sven/Desktop/temp.xlsx') as writer:
    for commentary_id in pd_commentaries:
        df = pd.DataFrame(sheet[commentary_id])
        df.to_excel(writer, sheet_name=commentary_id, index=False)


#%%
from text_importation.classes import Commentary

sheet = {'commentary': [], 'page': [], 'regions': [], 'word_count': []}

commentary = Commentary("sophoclesplaysa05campgoog", 'krakenhocr')


for page in commentary.pages:

    if int(page.id.split('_')[-1]) % 20 == 0:
        print("processing page  " + page.id)

    regions = [r for r in page.regions if r.region_type in ['introduction', 'preface', 'footnotes', 'commentary']]
    if not regions:
        continue
    else:
        sheet['commentary'].append(commentary.id)
        sheet['page'].append(page.id.split('_')[1])
        sheet['regions'].append(','.join([r.region_type for r in regions]))
        sheet['word_count'].append(sum([len(r.words) for r in regions]))



for x in sheet['word_count']:
    print(x)