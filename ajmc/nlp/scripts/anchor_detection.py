import re
from ajmc.text_processing.canonical_classes import CanonicalCommentary, CanonicalEntity

can = CanonicalCommentary.from_json(
    '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data/cu31924087948174/canonical/v2/28qmab_tess_base.json')

test_pages = [can.get_page('cu31924087948174_0022'),
              can.get_page('cu31924087948174_0064'),
              can.get_page('cu31924087948174_0089'),
              can.get_page('cu31924087948174_0167'),]

line_number_pattern = re.compile(r'\d{1,3}[\s\.]')

def get_entity_word_range(line, ent_start, ent_end):
    start = line.word_range[0]+len(line.text[:ent_start].split(' '))
    end = line.word_range[0]+len(line.text[:ent_end].split(' '))
    return start, end


for page in test_pages:
    for line in page.children.lines:
        match = line_number_pattern.match(line.text)
        if match:
            entity = CanonicalEntity(can,
                                     word_range=get_entity_word_range(line, match.start(), match.end()),
                                     shifts=(0,0),
                                     label='line_number',
                                     transcript=None,
                                     wikidata_id=None)
            page.children.entities.append(entity)

    page.draw_textcontainers(['entities'], f'/Users/sven/Desktop/{page.id}.png')

