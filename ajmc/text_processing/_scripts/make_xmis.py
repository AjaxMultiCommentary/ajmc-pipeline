from ajmc.commons import variables as vs
from ajmc.text_processing import cas_utils
from ajmc.text_processing.ocr_classes import OcrCommentary

# comm_ids = [
#     'Hermann1851',
#     'lestragdiesdeso00tourgoog',
#     'SchneidewinNauckRadermacher1913',
#     'cu31924087948174',
#     'sophokle1v3soph',
#     'sophoclesplaysa05campgoog',
#     'Wecklein1894',
#     'Finglass2011'
# ]

skiped = []
for comm_id in vs.ALL_COMM_IDS:
    comm = OcrCommentary.from_ajmc_data(id=comm_id, ocr_run='*tess_base')
    cas_utils.export_commentary_to_xmis(comm,
                                        make_jsons=True, make_xmis=True,

                                        region_types=['commentary'])
