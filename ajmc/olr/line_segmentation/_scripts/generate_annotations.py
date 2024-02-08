from ajmc.commons import variables as vs
from ajmc.olr.automatic_region_detection import write_csv_manually
from ajmc.text_processing.canonical_classes import CanonicalCommentary


# We select pages from canonical commentaries and export to via
# We export it to via
via_csv_dict = {k: v for k, v in vs.VIA_CSV_DICT_TEMPLATE.items()}

for comm_id in vs.ALL_COMM_IDS:
    comm = CanonicalCommentary.from_json(vs.get_comm_canonical_path_from_pattern(comm_id, '*tess_retrained'))
    for page in comm.olr_gt_pages:
        for i, line in enumerate(page.children.lines):
            via_csv_dict["filename"].append(str(page.image.path.relative_to(vs.AJMC_DATA_DIR)))
            via_csv_dict["file_size"].append(page.image.path.stat().st_size)
            via_csv_dict["file_attributes"].append("{}")
            via_csv_dict["region_count"].append(len(page.children.lines))
            via_csv_dict["region_id"].append(i)
            via_csv_dict["region_shape_attributes"].append({"name": "rect",
                                                            "x": line.bbox.xywh[0],
                                                            "y": line.bbox.xywh[1],
                                                            "width": line.bbox.xywh[2],
                                                            "height": line.bbox.xywh[3]})
            via_csv_dict["region_attributes"].append({"text": "line"})

write_csv_manually('test.csv', via_csv_dict, '/Users/sven/Desktop/')

# We re-ingest the lines into the canonical commentary, as following region ingestion

# We export the canonical commentary to alto

# We re-ingest the lines into the canonical commentary, as following region ingestion
