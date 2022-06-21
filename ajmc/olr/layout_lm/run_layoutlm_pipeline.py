from ajmc.olr.layout_lm.layoutlm import main

if __name__ == '__main__':
    from ajmc.olr.layout_lm.config import create_olr_config
    config = create_olr_config(
        json_path='/Users/sven/packages/ajmc/data/configs/simple_config_local.json'
    )
    main(config)