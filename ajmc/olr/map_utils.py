

metrics_abbrev = {'ap': 'AP', 'precision': 'P', 'recall': 'R', 'number': 'N'}


def initialize_general_results(ids_to_labels):
    general_results = {('info','exp'):[], ('all', 'mAP'):[]}
    general_results.update({(n, m):[] for n in ids_to_labels.values()
                            for m in metrics_abbrev.values()})
    return general_results


def update_general_results(general_results, metrics, xp_name, ids_to_labels):
    general_results[('info','exp')].append(xp_name)
    general_results[('all', 'mAP')].append(float(metrics['mAP']))

    for l_id, dict_ in metrics[0.5].items():
        general_results[(ids_to_labels[l_id], 'N')].append(dict_['precision'].shape[0])
        for m, score in dict_.items():
            general_results[(ids_to_labels[l_id], metrics_abbrev[m])].append(float(score) if m == 'ap' else float(score.mean()))

    return general_results