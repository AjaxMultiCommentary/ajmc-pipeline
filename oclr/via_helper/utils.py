import cv2
import numpy as np
import os
import csv


def correct_csv_manually(csv_filename, args):
    """manually corrects quoting error in output csv-files"""

    os.system("""sed -ia 's/ //g' """ + csv_filename)
    os.system("""sed -ia "s/'/ /g" """ + csv_filename)
    os.system("""sed -ia 's/ /""/g' """ + csv_filename)
    os.system("""sed -ia 's/},/}",/g' """ + csv_filename)
    os.system("""sed -ia 's/,{/,"{/g' """ + csv_filename)
    os.system("""sed -ia 's/""}/""}"/g' """ + csv_filename)


def write_csv_manually(csv_filename, csv_dict, args):
    """Writes a dictionnary a csv-file with custom quoting corresponding to via expectations"""

    pwd = os.getcwd()
    os.chdir(args.OUTPUT_DIR)

    with open(csv_filename, 'w') as csv_file:
        spamwriter = csv.writer(csv_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(list(csv_dict.keys()))
        for line in range(len(csv_dict["filename"])):
            to_append = []
            for k in list(csv_dict.keys()):
                to_append.append(csv_dict[k][line])
            spamwriter.writerow(to_append)

    correct_csv_manually(csv_filename, args)
    os.remove(csv_filename + "a")
    os.chdir(pwd)
