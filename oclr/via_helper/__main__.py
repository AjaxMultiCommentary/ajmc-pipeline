import os
from oclr.via_helper.zone_detector import detect_zones, merge_lace_and_detected_zones
from oclr.via_helper.utils import write_csv_manually
from oclr.utils.cli import args
from commons.variables import via_csv_dict_template

for filename in os.listdir(args.IMG_DIR):
    if filename[-3:] in ["png", "jpg", "tif", "jp2"]:
        print("Processing image " + filename)

        detect_zones(args, filename, via_csv_dict_template)
        # svg = file_management.ZonesMasks.from_lace_svg(args, filename[:-4], filename[-4:])
        # svg.convert_lace_svg_to_via_csv_dict_template(args)

# if args.merge_zones:
#     merged_zones = merge_lace_and_detected_zones(detected_zones, file_management.ZonesMasks.csv_dict, args)

write_csv_manually("detected_annotations.csv", via_csv_dict_template, args)
print("{} zones were automatically detected".format(len(via_csv_dict_template["filename"])))
