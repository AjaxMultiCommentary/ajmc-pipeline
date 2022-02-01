import csv
import json
import os
import Levenshtein
import sys
import numpy as np
from oclr.utils import file_management, extracters
from oclr.utils.cli import args
import oclr.evaluation.utils as eval_utils
import cv2


# TODO : Update Logging
# TODO : handle words that are in ocr but not in gt
# TODO : retrieve mean edit edit distance per word

def main(args, general_results, via_project):
    # ==================================== PRELIMINARY AFFECTATIONS ====================================================
    # Actualizing args
    args.ocr_engine = "ocrd" if ("ocr-d" in args.OCR_DIR.lower() or "ocrd" in args.OCR_DIR.lower()) else "lace"
    args.olr_annotation_type = "via" if args.via_project is not None else "lace"
    args.OCR_PARDIR = os.path.join(args.OCR_DIR, os.pardir)
    print("Processing " + args.OCR_PARDIR)

    if args.olr_annotation_type == "lace":
        args.zone_types = ["commentary", "primary_text", "translation", "app_crit", "page_number", "title", "no_zone"]
    else:
        args.zone_types = list(via_project["_via_attributes"]["region"]["text"]["options"].keys()) + ["no_zone"]
        args.zone_types.remove("undefined")

    if args.PARENT_DIR is not None:
        args.OUTPUT_DIR = os.path.join(args.OCR_PARDIR, "evaluation_fuzzy" if args.fuzzy_eval else "evaluation")
        os.makedirs(args.OUTPUT_DIR, exist_ok=True)

    # Pre-loop declarations
    error_counts = {i: {j: 0 for j in ["gt_chars", "distance", "gt_words", "false_words", "greek_chars", "numbers"]} for
                    i in
                    ["global"] + args.zone_types}

    error_counts["global"]["ocr_words"] = 0
    editops_record = []

    gt_wordlist = []
    ocr_wordlist = []
    # ==================================== LOOP OVER FILES IN GROUNDTRUTH ==================================================
    for gt_filename in os.listdir(args.GROUNDTRUTH_DIR):  # GT-files LOOP

        if gt_filename[-5:] == ".html":

            print("Evaluating file " + gt_filename)
            gt_filename = gt_filename[:-5]  # Removes the extension

            # Import image
            image = file_management.Image(args, gt_filename)

            # Import SVG or via .json
            if args.via_project is not None:
                zonemask = file_management.ZonesMasks.from_via_json(image, via_project)
            else:
                zonemask = file_management.ZonesMasks.from_lace_svg(args, image)

            groundtruth = file_management.OcrObject(args, image, gt_filename, data_type="groundtruth")
            ocr = file_management.OcrObject(args, image, gt_filename, data_type="ocr")

            gt_wordlist.append([w.content for w in groundtruth.words])
            ocr_wordlist.append([w.content for w in ocr.words])

            soup = eval_utils.initialize_soup(image)  # Initialize html output
            image.overlap_matrix = eval_utils.actualize_overlap_matrix(args, image, zonemask, groundtruth, ocr)

            # Loop over words in groundtruth
            for gt_word in groundtruth.words:
                gt_word.zone_type = extracters.get_segment_zonetype(args, gt_word, image.overlap_matrix)
                ocr_word = extracters.get_corresponding_ocr_word(args, gt_word, ocr, image.overlap_matrix)

                if args.fuzzy_eval:
                    gt_word.content = eval_utils.harmonise_unicode(gt_word.content)
                    ocr_word.content = eval_utils.harmonise_unicode(ocr_word.content)

                # Compute and record distance and edit operation
                distance = Levenshtein.distance(ocr_word.content, gt_word.content)
                editops = Levenshtein.editops(ocr_word.content, gt_word.content)
                editops_record = eval_utils.record_editops(gt_word, ocr_word, editops, editops_record)
                error_counts = eval_utils.actualize_error_counts(error_counts, gt_word, distance)

                # Actualize comparison files
                soup = eval_utils.insert_text(soup, image, gt_word, True, distance)
                soup = eval_utils.insert_text(soup, image, ocr_word, False, distance)

                # Draws word-boxes
                if args.draw_rectangles:
                    image.copy = eval_utils.draw_surrounding_rectangle(image.copy, gt_word, (0, 255, 0), 4)
                    image.copy = eval_utils.draw_surrounding_rectangle(image.copy, ocr_word, (0, 0, 255), 2)

            error_counts["global"]["ocr_words"] += len(ocr.words)
            # Write final html-file
            with open(os.path.join(args.OUTPUT_DIR, gt_filename + ".html"), "w") as html_file:
                html_file.write(str(soup))

            # Write boxes images
            if args.draw_rectangles:
                for zone in zonemask.zones:
                    image.copy = eval_utils.draw_surrounding_rectangle(image.copy, zone, (0, 0, 255), 2)
                    image.copy = eval_utils.draw_surrounding_rectangle(image.copy, zone, (0, 117, 117), 2, "raw")

                cv2.imwrite(os.path.join(args.OUTPUT_DIR, gt_filename + ".png"), image.copy)

    # ======================== COMPUTE METRICS : CER, WER AND MEAN EDIT-DISTANCE =====================================================
    cer = {}
    wer = {}
    med = {}
    for zone_type in error_counts.keys():

        try:
            cer[zone_type] = error_counts[zone_type]["distance"] / error_counts[zone_type]["gt_chars"]
        except ZeroDivisionError:
            cer[zone_type] = "NaN"

        try:
            wer[zone_type] = error_counts[zone_type]["false_words"] / error_counts[zone_type]["gt_words"]
        except ZeroDivisionError:
            wer[zone_type] = "NaN"

        try:
            med[zone_type] = error_counts[zone_type]["distance"] / error_counts[zone_type]["gt_words"]
        except ZeroDivisionError:
            med[zone_type] = "NaN"

    # error_counts["global"]["precision"], error_counts["global"]["recall"], error_counts["global"][
    #     "f1"] = eval_utils.compute_confusion_metrics(error_counts)

    # error_counts["global"]["precision"], error_counts["global"]["recall"], error_counts["global"][
    #     "f1"] = eval_utils.compute_confusion_metrics3(gt_wordlist, ocr_wordlist)

    prec = []
    rec = []
    f1 = []
    for gt, ocr in zip(gt_wordlist, ocr_wordlist):
        p, r, f = eval_utils.compute_confusion_metrics3(gt, ocr)
        prec.append([p, len(gt)])
        rec.append([r, len(gt)])
        f1.append([f,len(gt)])


    # TODO : See if the try/except is necessary
    try :
        error_counts["global"]["precision"] = sum([l[0]*l[1] for l in prec])/sum([l[1] for l in prec])
    except ZeroDivisionError:
        error_counts["global"]["precision"] = np.nan
    try:
        error_counts["global"]["recall"] = sum([l[0]*l[1] for l in rec])/sum([l[1] for l in rec])
    except ZeroDivisionError:
        error_counts["global"]["recall"] = np.nan
    try:
        error_counts["global"]["f1"] = sum([l[0]*l[1] for l in f1])/sum([l[1] for l in f1])
    except ZeroDivisionError:
        error_counts["global"]["f1"] = np.nan





    # =================================== WRITE EDIT-OPERATIONS RECORD =====================================================
    editops_record = {op: editops_record.count(op) for op in editops_record}
    editops_record = {k: v for k, v in sorted(editops_record.items(), key=lambda item: item[1], reverse=True)}

    with open(os.path.join(args.OUTPUT_DIR, "editops_record.tsv"), 'w') as csv_file:
        spamwriter = csv.writer(csv_file, delimiter='\t', quotechar='"')
        for k, v in editops_record.items():
            spamwriter.writerow([k, v])

    # ======================================== WRITE RESULTS .TSV FILE =====================================================
    with open(os.path.join(args.OUTPUT_DIR, "results_direct_average.tsv"), 'w') as csv_file:
        spamwriter = csv.writer(csv_file, delimiter='\t', quotechar='"')

        header1 = []
        header2 = []
        counts = []
        stats = []

        for key in error_counts.keys():
            header1.append(key)
            header1.append(key)
            header1.append(key)
            header2.append("cer")
            header2.append("wer")
            header2.append("med")
            counts.append(str(error_counts[key]["gt_chars"]))
            counts.append(str(error_counts[key]["gt_words"]))
            counts.append(str(error_counts[key]["gt_words"]))
            stats.append(cer[key])
            stats.append(wer[key])
            stats.append(med[key])

        for key in ["f1", "recall", "precision"]:
            header1.insert(0, "global")
            header2.insert(0, key)
            counts.insert(0, "-")
            stats.insert(0, error_counts["global"][key])

        spamwriter.writerow(header1)
        spamwriter.writerow(counts)
        spamwriter.writerow(stats)

        if args.PARENT_DIR is not None:
            stats.insert(0, args.OCR_DIR.split("/")[-2])
            stats.insert(0, args.PARENT_DIR.split("/")[-1])

            if general_results == []:
                general_results.append(["", ""] + header1)
                general_results.append(["", ""] + header2)

            general_results.append(stats)

    # ======================================== WRITE COUNTS .TSV FILE =====================================================
    with open(os.path.join(args.OUTPUT_DIR, "stats.tsv"), 'w') as csv_file:
        spamwriter = csv.writer(csv_file, delimiter='\t', quotechar='"')

        header1 = []
        header2 = []
        counts = []
        subkeys = ["chars", "words", "greek_chars", "greek_chars_%", "numbers"]

        for key in error_counts.keys():
            header1 += [key] * len(subkeys)
            header2 += subkeys

            counts.append(str(error_counts[key]["gt_chars"]))
            counts.append(str(error_counts[key]["gt_words"]))
            counts.append(str(error_counts[key]["greek_chars"]))
            try:
                counts.append(str(100 * error_counts[key]["greek_chars"] / error_counts[key]["gt_chars"]))
            except ZeroDivisionError:
                counts.append("NaN")
            counts.append(str(error_counts[key]["numbers"]))

        spamwriter.writerow(header1)
        spamwriter.writerow(header2)
        spamwriter.writerow(counts)


# ======================================== WRITE GENERAL RESULTS AND RUN ===============================================

if args.PARENT_DIR is not None:
    args.PARENT_DIR_STORED = args.PARENT_DIR[:]

    # for commentary_name in ["campbell", "jebb", "lobeck", "schneidewin", "wecklein"]:
    for commentary_name in ["lobeck"]:
        general_results = []
        args.PARENT_DIR = os.path.join(args.PARENT_DIR_STORED, commentary_name)
        args.IMG_DIR = os.path.join(args.PARENT_DIR, "ocr/evaluation/groundtruth/images")
        args.GROUNDTRUTH_DIR = os.path.join(args.PARENT_DIR, "ocr/evaluation/groundtruth/html")
        args.via_project = os.path.join(args.PARENT_DIR, "olr/via_project.json")
        with open(args.via_project, "r") as file:
            via_project = json.load(file)

        for dir in os.scandir(os.path.join(args.PARENT_DIR, "ocr/ocrs")):
            if dir.is_dir():
                args.OCR_DIR = os.path.join(dir.path, "outputs")
                main(args, general_results, via_project)

        with open(os.path.join(
                args.PARENT_DIR,
                "ocr/evaluation/general_results_fuzzy.tsv" if args.fuzzy_eval else "ocr/evaluation/general_results_direct_average.tsv"),
                'w') as csv_file:
            spamwriter = csv.writer(csv_file, delimiter='\t', quotechar='"')
            for el in general_results:
                spamwriter.writerow(el)


else:
    with open(args.via_project, "r") as file:
        via_project = json.load(file)
    main(args, None, via_project)
