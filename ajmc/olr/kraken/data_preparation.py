from typing import List, Optional

from bs4 import BeautifulSoup

from ajmc.commons import variables
from ajmc.commons.miscellaneous import read_google_sheet
from ajmc.text_importation.classes import OcrCommentary


def get_olr_split_pages(commentary: 'TextContainer',
                        splits: List[str]) -> List['TextContainer']:
    """Gets the data from splits on the olr_gt sheet."""

    olr_gt = read_google_sheet(variables.SPREADSHEETS_IDS['olr_gt'], 'olr_gt')

    filter_ = [(olr_gt['commentary_id'][i] == commentary.id and olr_gt['split'][i] in splits) for i in
               range(len(olr_gt['page_id']))]

    return [p for p in commentary.pages if p.id in list(olr_gt['page_id'][filter_])]


def create_kraken_alto(img_path: str,
                       regions: List['TextContainer'],
                       output_path: Optional[str] = None):
    base = """<?xml version="1.0" encoding="UTF-8">
    <alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
    xmlns="http://www.loc.gov/standards/alto/ns-v4#"
    xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-0.xsd">
    </alto>"""

    soup = BeautifulSoup(base, 'xml')
    soup.alto.append(soup.new_tag('Description'))
    img_path_tag = soup.new_tag('sourceImageInformation')
    img_path_tag.string = img_path
    soup.alto.Description.append(img_path_tag)


    printsp = soup.new_tag('PrintSpace')
    page = soup.new_tag('Page')
    layout = soup.new_tag('Layout')
    page.append(printsp)
    layout.append(page)
    soup.alto.append(layout)

    for r in regions:
        reg = soup.new_tag('ComposedBlockType',
                           ID=r.id,

                           HPOS=r.coords[0].xywh[0],
                           VPOS=r.coords[0].xywh[1],
                           WIDTH=r.coords[0].xywh[2],
                           HEIGHT=r.coords[0].xywh[3],
                           TYPE=r.info['region_type'])

        soup.alto.Layout.Page.PrintSpace.append(reg)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(soup.prettify())

    return soup

ocr_comm = OcrCommentary.from_ajmc_structure("/Users/sven/packages/ajmc/data/sample_commentaries/cu31924087948174/ocr/runs/tess_eng_grc/outputs")
can_comm = ocr_comm.to_canonical()




for p in can_comm.children['page']:
    create_kraken_alto(img_path=p.images[0].path,
                       regions=p.children['region'],
                       output_path=f'/Users/sven/Desktop/{p.id}.xml')