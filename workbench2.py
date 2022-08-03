import os

from ajmc.text_processing.ocr_classes import OcrCommentary
from ajmc.text_processing.canonical_classes import CanonicalCommentary


can = OcrCommentary.from_ajmc_structure('/Users/sven/packages/ajmc/data/sample_commentaries/cu31924087948174/ocr/runs/tess_eng_grc/outputs').to_canonical()
can.to_json('/Users/sven/packages/ajmc/data/sample_commentaries/cu31924087948174/canonical/v2/tess_eng_grc.json')
can_from_json = CanonicalCommentary.from_json('/Users/sven/packages/ajmc/data/sample_commentaries/cu31924087948174/canonical/v2/tess_eng_grc.json')

for p in can_from_json.children['page']:
    p.to_alto(['region', 'line'], os.path.join('/Users/sven/drive/ketos', p.id+'.xml'))



