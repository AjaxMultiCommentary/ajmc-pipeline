import os
from olr.region_detection.utils import write_csv_manually, detect_regions
from utils.cli import args
from commons.variables import via_csv_dict_template



for filename in os.listdir(args.img_dir):
    if filename[-3:] in ["png", "jpg", "tif", "jp2"]:
        print("Processing image " + filename)

        detect_regions(img_path=os.path.join(args.img_dir, filename),
                       output_dir=args.output_dir,
                       dilation_kernel_size=args.dilation_kernel_size,
                       dilation_iterations=args.dilation_iterations,
                       artifact_size_threshold=args.artifact_size_threshold,
                       draw_rectangles=args.draw_rectangles,
                       via_csv_dict=via_csv_dict_template)

write_csv_manually("detected_annotations.csv", via_csv_dict_template, args.output_dir)
print("{} zones were automatically detected".format(len(via_csv_dict_template["filename"])))
