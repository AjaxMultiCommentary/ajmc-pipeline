# %% Imports and defs
import pandas as pd
import numpy as np
import re


def weighted_average_series(series: list, weights: list):
    """
    Computes weighted averages with NaN consideration
    :param series: a list of pandas series
    :param weights: a list of pandas series
    :return:
    :rtype:
    """

    sumprod = series[0][:]
    sumprod[:] = 0
    weights_sum = 0

    for i in range(len(series)):
        values = series[i].fillna(0)
        weights = weights[i].fillna(0)
        weights_sum += np.sum(weights)
        temp = values * weights
        sumprod += temp

    def safe_div(x, y):
        return x / y if y != 0 else np.nan

    for i in range(len(sumprod)):
        sumprod.iloc[i] = safe_div(sumprod.iloc[i], weights_sum)


def weight_average(values, weights):
    """Computes the weighted average of a series
    :param values: A pandas series of values
    :param weights: A pandas series of corresponding weights
    :return: a number or a nan
    """
    values = values.fillna(0)
    weights = weights.fillna(0)
    return np.sum(values.fillna(0) * weights.fillna(0)) / np.sum(weights) if np.sum(weights) != 0 else np.nan


# %% Import results
results = pd.DataFrame()
for commentary_name in ["campbell", "jebb", "lobeck", "schneidewin", "wecklein"]:
    path = "/Users/sven/Google Drive/_AJAX/AjaxMultiCommentary/data/commentary_data/" + \
           commentary_name + "/ocr/evaluation/general_results.tsv"

    # results[commentary_name] = pd.read_csv(path, sep="\t", header=[0, 1], index_col=[0,1])
    results = results.append(pd.read_csv(path, sep="\t", header=[0, 1], index_col=[0, 1]))

results.index.names = pd.Index(["commentary", "pipeline"])
results.columns.names = pd.Index(["region", "metric"])

for el in results.index:
    print(el)

results.loc[("campbell", "lace_best"), :] = results.loc[("campbell",
                                                         "lace_base_cu31924087948174-2021-05-23-21-05-58-porson-2021-05-23-14-27-27"),
                                            :]
results.loc[("jebb", "lace_best"), :] = results.loc[("jebb",
                                                     "lace_base_sophoclesplaysa05campgoog-2021-05-23-21-38-49-porson-2021-05-23-14-27-27"),
                                        :]
results.loc[("lobeck", "lace_best"), :] = results.loc[("lobeck", "lace_retrained"), :]
results.loc[("schneidewin", "lace_best"), :] = results.loc[("schneidewin", "lace_retrained"), :]
results.loc[("wecklein", "lace_best"), :] = results.loc[("wecklein", "lace_base"), :]

# %% Import counts
counts = pd.DataFrame()
for commentary_name in ["campbell", "jebb", "lobeck", "schneidewin", "wecklein"]:
    path = "/Users/sven/Google Drive/_AJAX/AjaxMultiCommentary/data/commentary_data/" + \
           commentary_name + "/ocr/ocrs/ocrd_min/evaluation/stats.tsv"

    # results[commentary_name] = pd.read_csv(path, sep="\t", header=[0, 1], index_col=[0,1])
    temp = pd.read_csv(path, sep="\t", header=[0, 1])
    temp.set_index(pd.Index([commentary_name]), inplace=True)

    counts = counts.append(temp)

# %% Make table 1: regionized scores

# Define parameters
selected_pipelines = ["lace_best", "ocrd_min", "ocrd_vanilla"]
selected_regions = {
    "global": ["global"],
    "primary_text": ["primary_text"],
    "commentary-like": ["commentary", "footnote"],
    "low_greek": ["introduction", "preface", "translation"],
    "app_crit": ["app_crit"],
    "structured_texts": ["appendix", "bibliography", "index_siglorum", "running_header", "table_of_contents", "title"],
    "numbers": ["line_number_commentary", "line_number_text", "page_number"],
}
selected_metrics = ["cer", "wer"]
selected_stats = ["mean", "std"]

# %% weight-average models over commentaries and their respective number of caracters
avg = pd.DataFrame()
for pipeline in selected_pipelines:
    temp = results.xs(pipeline, level="pipeline")
    l = []
    for column in temp:
        l.append(weight_average(temp.loc[:, column], counts.loc[:, (column[0], "chars")]))
        l.append(np.std(temp.loc[:, column]))

    # Create the new def
    multiindex_temp = []
    for el in temp.columns:
        for stat in [tuple(("mean",)), tuple(("std",))]:
            multiindex_temp.append(el + stat)

    multiindex_temp = pd.MultiIndex.from_tuples(multiindex_temp, names=["region", "metric", "stats"])

    avg_temp = pd.DataFrame(l, columns=pd.Index([pipeline]), index=multiindex_temp).T
    avg = avg.append(avg_temp)

# %% weight-average grouped regions modulo the number of chars they have

# add columns
for item in selected_regions.items():  # All this should be much faster with pd.df.groupby
    if len(item[1]) > 1:  # Don't do anything for groups containing only one column
        for metric in selected_metrics:
            for stat in selected_stats:
                stat_temp = []
                for pipeline in selected_pipelines:
                    values = avg.loc[pipeline, item[1]].xs(metric, level="metric").xs(stat,
                                                                                      level="stats")  # returns a Series
                    key = "chars" if metric == "cer" else "words"
                    weights = np.sum(counts).loc[item[1], key]
                    stat_temp.append(weight_average(values, weights.xs(key, level=1)))
                avg[(item[0], metric, stat)] = stat_temp

# filter dataset
filter_index = []
for item in selected_regions.items():
    if item[0] == "global":
        filter_index += [(item[0], metric, stat) for metric in ["f1", "cer", "wer"] for stat in
                         selected_stats]
    else:
        for stat in selected_stats:
            filter_index.append((item[0], "cer", stat))

avg_filtered = avg.loc[:, filter_index]

# %%add numbers of greek chars
counts_row = {}
for item in selected_regions.items():
    chars = np.sum(counts).loc[item[1], "chars"]
    chars = int(np.sum(chars))
    greek_chars = np.sum(counts).loc[item[1], "greek_chars"]
    greek_chars = int(np.sum(greek_chars))
    percentage = round(100 * greek_chars / chars)
    counts_row[item[0]] = str(chars) + " ({}%)".format(str(percentage))


# %% Build Export-ready dataset

def merge_mean_and_std(number1, number2):
    number1 = str(round(number1, ndigits=2))
    number1 = re.sub(r"0\.", ".", number1)
    number2 = str(round(number2, ndigits=2))
    number2 = re.sub(r"0\.", ".", number2)
    return number1 + "±" + number2


mean_df = avg_filtered.xs("mean", level="stats", axis=1)
std_df = avg_filtered.xs("std", level="stats", axis=1)

export_df = avg_filtered.xs("mean", level="stats", axis=1)

# Merge index with counts
for column in mean_df:
    export_df.loc[:, column] = [merge_mean_and_std(mean_df.loc[:, column][i], std_df.loc[:, column][i]) for i in
                                range(len(mean_df.loc[:, column]))]

#%%
# Build a new column index with counts
new_index = []
for column in export_df:
    new_index.append((column[0].capitalize(), counts_row[column[0]], column[1].upper()))
export_df.columns = pd.MultiIndex.from_tuples(new_index, names=["Region", "Char counts", "Metric"])


# Pretify export df

export_df.to_latex(buf="/Users/sven/Desktop/test.txt")













# %% Make commentary table
avg_temp = results[[("global", "f1"), ("global", "cer")]]
index_names = list(set([i[0] for i in avg_temp.index.values]))
filter = [(i, j) for i in sorted(index_names) for j in ["ocrd_vanilla", "ocrd_min", "lace_best"]]
avg_temp = avg_temp.loc[filter]
avg_temp = pd.concat(
    [avg_temp.loc[commentary_name].rename(columns={"global": commentary_name}) for commentary_name in
     sorted(index_names)],
    axis=1)

# Adding NLD
for column in avg_temp:
    if column[1] == "cer":
        avg_temp.loc[:, (column[0], "NLD")] = 1 - avg_temp.loc[:, column]

# Final processes

# Sorting
final = pd.concat([
    avg_temp.xs("lobeck", axis=1, level="region", drop_level=False),
    avg_temp.xs("schneidewin", axis=1, level="region", drop_level=False),
    avg_temp.xs("campbell", axis=1, level="region", drop_level=False),
    avg_temp.xs("jebb", axis=1, level="region", drop_level=False),
    avg_temp.xs("wecklein", axis=1, level="region", drop_level=False),
], axis=1)

#changing names
new_colnames = [(column[0].capitalize(), column[1].upper()) for column in final]
final.columns = pd.MultiIndex.from_tuples(new_colnames, names=["Commentary", "Metric"])
final.index = pd.Index(["Calamari GT4Hist", "Tesseract", "Kraken+Ciaconna"])

# %%
final.to_latex(buf="/Users/sven/Desktop/test2.txt", float_format="%.2f")
