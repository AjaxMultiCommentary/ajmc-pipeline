from pathlib import Path

from ajmc.nlp.token_classification.pipeline import main

base_config = {
    "do_train": True,
    "do_seqeval": True,
    "overwrite_outputs": True,
    "device_name": "cuda:1",
    "epochs": 40,
    "batch_size": 4,
    "evaluate_during_training": True,
    # "model_max_length": 512,
}
model_names_and_paths = [
    ('dbmdz/bert-base-historic-multilingual-cased', None),
    ('bowphs/PhilBerta', None),
    ('FacebookAI/xlm-roberta-base', None),
    ('google/canine-c', None),
    ('google/canine-c', '/scratch/sven/canine/output/checkpoint-25000'),
    ('google/canine-c', '/scratch/sven/canine/output/checkpoint-50000'),
    ('google/canine-c', '/scratch/sven/canine/output/checkpoint-75000'),
    ('google/canine-c', '/scratch/sven/canine/output/checkpoint-100000'),
    ('google/canine-c', '/scratch/sven/canine/output/checkpoint-125000'),
    ('xlm-roberta-base', '/scratch/sven/xlm-roberta-base/outputs/checkpoint-5000'),
    ('xlm-roberta-base', '/scratch/sven/xlm-roberta-base/outputs/checkpoint-10000'),
    ('xlm-roberta-base', '/scratch/sven/xlm-roberta-base/outputs/checkpoint-15000'),
    ('xlm-roberta-base', '/scratch/sven/xlm-roberta-base/outputs/checkpoint-20000'),

]

train_paths = {'lemlink': '/scratch/sven/ajmc_data/lemma-linkage-corpus/data/release/v1.0.alpha/lemlink-v1.0.alpha-train_NOCOMMENT.tsv',
               'ner': '/scratch/sven/ajmc_data/AjMC-NE-corpus/data/release/v0.4/ajmc-v0.4-train-en.tsv'}

eval_paths = {'lemlink': '/scratch/sven/ajmc_data/lemma-linkage-corpus/data/release/v1.0.alpha/lemlink-v1.0.alpha-test_NOCOMMENT.tsv',
              'ner': '/scratch/sven/ajmc_data/AjMC-NE-corpus/data/release/v0.4/ajmc-v0.4-test-en.tsv'}

labels_columns = {'lemlink': 'LABEL',
                  'ner': 'NE-COARSE-LIT'}

main_output_dir = Path('/scratch/sven/lm_benchmark_exps/')


def get_full_model_name(model_name, model_path):
    model_full_name = f'{model_name.split("/")[-1]}'
    if model_path is not None:
        model_full_name += f'_{model_path.split("/")[-1]}'
    return model_full_name


def get_model_output_dir(model_name, model_path, data_format) -> Path:
    return main_output_dir / get_full_model_name(model_name, model_path) / data_format


#%%

for model_name, model_path in model_names_and_paths:
    print(f'Running experiments for model {model_name}')
    for data_format in ['lemlink', 'ner']:
        print(f'Running experiments for data format {data_format}')
        config = base_config.copy()

        config['output_dir'] = str(get_model_output_dir(model_name, model_path, data_format))

        # if the output directory already exists and is not empty and we are not overwriting, skip
        if not config['overwrite_outputs'] and (Path(config['output_dir']) / 'results/seqeval/best_results.tsv').exists():
            print(f'Skipping {model_name} - {data_format} as output directory already exists and is not empty')
            continue

        config['model_name'] = model_name
        config['model_path'] = model_path
        config['data_format'] = data_format
        config['train_path'] = train_paths[data_format]
        config['eval_path'] = eval_paths[data_format]
        config['labels_column'] = labels_columns[data_format]

        if 'canine' in config['model_name']:
            config['model_max_length'] = 2048
        else:
            config['model_max_length'] = 512

        main(config_dict=config)

#%% We now generate a general results table for the experiments we just ran.
import pandas as pd

# start by reading the best_results.tsv in each subdirectory

results = []
for model_name, model_path in model_names_and_paths:
    full_model_name = get_full_model_name(model_name, model_path)
    for data_format in ['lemlink', 'ner']:
        output_dir = get_model_output_dir(model_name, model_path, data_format)
        best_results = pd.read_csv(output_dir / 'results/seqeval/best_results.tsv', sep='\t', header=[0, 1])
        best_results[('exp', 'model')] = full_model_name
        best_results[('exp', 'data_format')] = data_format
        results.append(best_results)

results_df = pd.concat(results)

columns_to_keep = [('exp', 'model'), ('exp', 'data_format'), ('ALL', 'F1'), ('ALL', 'A'), ('ALL', 'P'), ('ALL', 'R')]
results_df = results_df[columns_to_keep]

results_df.to_csv(main_output_dir / 'results_df.tsv', sep='\t', index=False)

#%% We now generate plots with the training results for each model and data format
import matplotlib.pyplot as plt


# Create a big single figure with all the plots of training loss and F1 score for each model and data format
fig, axs = plt.subplots(int(len(model_names_and_paths)), 2, figsize=(12, 36))

for i, (model_name, model_path) in enumerate(model_names_and_paths):
    for j, data_format in enumerate(['lemlink', 'ner']):
        output_dir = get_model_output_dir(model_name, model_path, data_format)
        training_results = pd.read_csv(output_dir / 'results/seqeval/train_results.tsv', sep='\t', header=[0, 1])
        axs[i, j].plot(training_results[('TRAINING', 'EP')], training_results[('TRAINING', 'LOSS')], label='train_loss')
        axs[i, j].plot(training_results[('TRAINING', 'EP')], training_results[('ALL', 'F1')], label='eval_f1')
        axs[i, j].set_title(f'{get_full_model_name(model_name, model_path)} - {data_format}')
        # constrain the y-axis to 0-1
        axs[i, j].set_ylim(0, 1)
        axs[i, j].legend()

plt.tight_layout()

# save the figure
fig.savefig(main_output_dir / 'training_results.png')
