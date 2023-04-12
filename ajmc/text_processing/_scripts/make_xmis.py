from ajmc.commons import variables as vs
from ajmc.text_processing import cas_utils
from ajmc.text_processing.ocr_classes import OcrCommentary

comm_ids = [
    # 'Finglass2011',
    'Hermann1851',
    # 'lestragdiesdeso00tourgoog',
    'SchneidewinNauckRadermacher1913',
    # 'cu31924087948174',
    # 'sophokle1v3soph',
    # 'sophoclesplaysa05campgoog',
    # 'Wecklein1894',
    'Stanford1963',

]

skiped = []
for comm_id in comm_ids:
    comm = OcrCommentary.from_ajmc_data(id=comm_id, ocr_run_id='*tess_base')

    cas_utils.export_commentary_to_xmis(comm,
                                        make_jsons=True,
                                        make_xmis=True,
                                        jsons_dir=vs.get_comm_lemlink_jsons_dir(comm.id, comm.ocr_run_id),
                                        xmis_dir=vs.get_comm_lemlink_xmis_dir(comm.id, comm.ocr_run_id),
                                        region_types=['commentary'],
                                        overwrite=False, )
