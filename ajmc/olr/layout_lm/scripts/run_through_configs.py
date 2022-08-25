import json
import os

configs_dir = 'data/configs/layoutlm/configs'

prefix = '/content/drive/MyDrive/'
output_dir = os.path.join(prefix, 'layout_lm_tests')

for fname in os.listdir(configs_dir):
    if fname.endswith('.json'):
        with open(os.path.join(configs_dir, fname), "r") as file:
            config = json.loads(file.read())

        config['output_dir'] = os.path.join(output_dir, fname[:-5])
        config['batch_size'] = 8
        config["device_name"] = "cuda"
        config["do_debug"] = False
        config["do_save"] = True
        config["do_train"] = True
        config["do_draw"] = False
        config["epochs"] = 50
        config["evaluate_during_training"] = True
        config["gradient_accumulation_steps"] = 2
        config["overwrite_output_dir"] = True

        with open(os.path.join(configs_dir, fname), "w") as outfile:
            json.dump(config, outfile, indent=4, ensure_ascii=False, sort_keys=True)