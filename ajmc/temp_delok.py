from ajmc.olr.layout_lm.layoutlm import main
from ajmc.olr.layout_lm.config import create_olr_config
import os
from ajmc.commons.variables import PATHS
from ajmc.commons.miscellaneous import stream_handler

stream_handler.setLevel(0)
config_dir = "/scratch/sven/tmp/ajmc/data/layoutlm/configs"


for fname in sorted(os.listdir(config_dir)):
    if fname[:2] in ['4A']:
        print("----- Running config :   ", fname)
        config = create_olr_config(os.path.join(config_dir, fname), prefix=PATHS['cluster_base_dir'])
        config['output_dir'] = "/scratch/sven/4A_EXP"
        main(config)