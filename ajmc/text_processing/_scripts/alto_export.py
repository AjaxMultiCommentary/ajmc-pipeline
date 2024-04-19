from ajmc.commons import variables as vs
from ajmc.text_processing.canonical_classes import CanonicalCommentary

alto_dir = '/Users/sven/Desktop/coucou'
comm_id = 'sophokle1v3soph'
json_path = vs.get_comm_canonical_path_from_ocr_run_pattern(comm_id, '*_tess_base')
canonical_commentary = CanonicalCommentary.from_json(json_path)
canonical_commentary.to_alto(output_dir=alto_dir,
                             children_types=['regions', 'lines'],
                             region_types_mapping=vs.REGION_TYPES_TO_SEGMONTO,
                             region_types_ids=vs.SEGMONTO_TO_VALUE_IDS,
                             copy_images=True)
