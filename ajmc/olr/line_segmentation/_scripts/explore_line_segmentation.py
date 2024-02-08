# Find commentaries with kraken runs and note comm_ids and run_ids
from ajmc.commons import variables as vs

comm_ids = []
kraken_run_ids = []
tess_run_ids = []

for comm_id in ['sophokle1v3soph']:
    comm_runs_dir = vs.get_comm_ocr_runs_dir(comm_id)
    lace_base_run_id = [p.name for p in comm_runs_dir.glob('*lace_base')]
    if lace_base_run_id:
        lace_base_run_id = lace_base_run_id[0]
        comm_ids.append(comm_id)
        kraken_run_ids.append(lace_base_run_id)
        tess_run_ids.append([p.name for p in comm_runs_dir.glob('*tess_retrained')][0])

#%% For each commentary, get the ocr commentary and select N pages.
from ajmc.text_processing.ocr_classes import OcrCommentary
from pathlib import Path

page_ids = Path('/Users/sven/Desktop/page_ids.txt').read_text().split('\n')

for comm_id, kraken_run_id, tess_run_id in zip(comm_ids, kraken_run_ids, tess_run_ids):
    kraken_comm = OcrCommentary.from_ajmc_data(comm_id, ocr_run_id=kraken_run_id)
    # tess_comm = OcrCommentary.from_ajmc_data(comm_id, ocr_run_id=tess_run_id)
    # comm_json_path = vs.get_comm_canonical_default_path(comm_id, tess_run_id)
    # canonical_comm = CanonicalCommentary.from_json(comm_json_path)

    # page_ids = random.sample([p.id for p in kraken_comm.children.pages], 20)
    kraken_pages = [p for p in kraken_comm.children.pages if p.id in page_ids]
    # tess_pages = [p for p in tess_comm.children.pages if p.id in page_ids]
    # canonical_pages = [p for p in canonical_comm.children.pages if p.id in page_ids]

    for page in kraken_pages:
        output_path = Path('/Users/sven/Desktop/kraken_pages/') / f'{page.id}.png'
        page.draw_textcontainers(tc_types=['lines'], output_path=output_path)
