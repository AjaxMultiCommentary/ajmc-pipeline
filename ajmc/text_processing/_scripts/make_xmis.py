from ajmc.commons import variables as vs
from ajmc.text_processing import cas_utils
from ajmc.text_processing.raw_classes import RawCommentary

comm_ids = [
    'Ferrari1974',
    'Garvie1998',
    'lestragdiesdeso00tourgoog',
    'Paduano1982',
    'pvergiliusmaroa00virggoog',
    'thukydides02thuc',
    'Untersteiner1934',
]

for comm_id in comm_ids:
    comm = RawCommentary.from_ajmc_data(id=comm_id, ocr_run_id='*tess_base')

    cas_utils.export_commentary_to_xmis(comm,
                                        make_jsons=True,
                                        make_xmis=True,
                                        jsons_dir=vs.get_comm_lemlink_jsons_dir(comm.id, comm.ocr_run_id),
                                        xmis_dir=vs.get_comm_lemlink_xmis_dir(comm.id, comm.ocr_run_id),
                                        region_types=['commentary'],
                                        overwrite=False, )
