import os
from tqdm import tqdm
from ajmc.ocr.preprocessing.data_preparation import resize_ocr_dataset
from ajmc.ocr.tesseract.tesseract_utils import reformulate_output_dir, run_tesseract, evaluate_tesseract


ziqis_models = '/Users/sven/Desktop/tess_xps/models'
for model in tqdm(os.listdir(ziqis_models)):
    if model.endswith('.traineddata'):
        model = model[:-len('.traineddata')]
        for size in [30, 70, None]:
            if size is None:
                data_dir = f'/Users/sven/Desktop/tess_xps/data/ajmc_gr/ajmc_gr_lines'
                output_dir = f'/Users/sven/Desktop/tess_xps/results/{model}_gr_lines'
                output_dir = reformulate_output_dir(output_dir)

            else:
                data_dir = f'/Users/sven/Desktop/tess_xps/data/ajmc_gr/ajmc_gr_lines_rsz{size}'
                if not os.path.exists(data_dir):
                    resize_ocr_dataset(dataset_dir='/Users/sven/Desktop/tess_xps/data/ajmc_gr/ajmc_gr_lines',
                                       output_dir=data_dir,
                                       target_height=size)

                output_dir = f'/Users/sven/Desktop/tess_xps/results/{model}_gr_lines_rsz{size}'
                output_dir = reformulate_output_dir(output_dir)

            run_tesseract(img_dir=data_dir,
                          output_dir=str(output_dir),
                          langs=model,
                          psm=7,
                          tessdata_prefix= ziqis_models)

            evaluate_tesseract(gt_dir=data_dir,
                               ocr_dir=str(output_dir))
