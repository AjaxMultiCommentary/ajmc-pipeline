import os
import csv

base_dir = "../exps/evaluation"
commentaries = ["cu31924087948174", "sophokle1v3soph"]

def move_to_top(model_str, replace_l):
    for scheme in replace_l:
        if model_str.endswith(scheme):
            print(model_str, scheme)
            return scheme[1:] + "+" + model_str.replace(scheme, "")
    return model_str

def detect_sort_key(model_str):
    for scheme in replace_lists:
        if model_str.endswith(scheme):
            print(model_str, scheme)
            if scheme.startswith("+"):
                return "*" + scheme
            return scheme
    for scheme in start_lists:
        if model_str.startswith(scheme):
            print(model_str, scheme)
            if scheme.endswith("+"):
                return scheme + "*"
            return scheme
    
    return "self"

start_lists = [
    "grc+eng+GT4HistOCR_50000000.997_191951+",
    "grc+deu+GT4HistOCR_50000000.997_191951+",
    "grc+eng+GT4HistOCR_50000000.997_191951",
    "grc+deu+GT4HistOCR_50000000.997_191951",
    "grc+GT4HistOCR_50000000.997_191951+",
    "grc+GT4HistOCR_50000000.997_191951",
    "grc+eng+",
    "grc+eng",
    "grc+deu+",
    "grc+deu",
    "grc+",
    "grc"
]

end_lists = [
    "+deu+GT4HistOCR_50000000.997_191951",
    "+eng+GT4HistOCR_50000000.997_191951",
    "deu+GT4HistOCR_50000000.997_191951",
    "eng+GT4HistOCR_50000000.997_191951",
    "+GT4HistOCR_50000000.997_191951",
    "GT4HistOCR_50000000.997_191951",
    "+eng",
    "+deu",
]

replace_lists = [
    "+eng+GT4HistOCR_50000000.997_191951",
    "+deu+GT4HistOCR_50000000.997_191951",
    "+GT4HistOCR_50000000.997_191951", 
    "+eng", 
    "+deu",
]

for commentary in commentaries:
    aggregated_file = os.path.join(base_dir, f"{commentary}.tsv")
    run_dir = os.path.join(base_dir, commentary, "ocr", "runs")
    initial = True
    with open(aggregated_file, "w") as f_out:
        writer = csv.writer(f_out, delimiter='\t')
        rows = []
        for run_id in os.listdir(run_dir):
            performance_file = os.path.join(run_dir, run_id, "evaluation", "evaluation_results.tsv")
            if not os.path.isfile(performance_file):
                print(f"Currently no evaluation result for {run_id}")
                continue
            with open(performance_file, "r") as f_in:
                reader = csv.reader(f_in, delimiter="\t")
                for n in range(3):
                    row = next(reader)
                    if initial:
                        row.insert(0, "sort_key")
                        row.insert(0, "run_id")
                        # print(row)
                        writer.writerow(row)
                    else:
                        continue
                initial=False
                for row in reader:
                    row.insert(0, detect_sort_key(run_id.split("_", 2)[-1]))
                    row.insert(0, move_to_top(run_id.split("_", 2)[-1], replace_lists))
                    rows.append(row)
        rows.sort(key=lambda x: x[0])
        writer.writerows(rows)