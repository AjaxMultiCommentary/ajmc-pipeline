import os

base_dir = '/Users/sven/drive/_AJAX/AjaxMultiCommentary/data/commentaries/commentaries_data'


for comm_dir in next(os.walk(base_dir))[1]:
    comm_runs_dir = os.path.join(base_dir, comm_dir, 'ocr/runs/')
    if os.path.isdir(comm_runs_dir):
        for run_name in next(os.walk(comm_runs_dir))[1]:
            outputs_dir = os.path.join(comm_runs_dir, run_name, 'outputs')
            for fname in os.listdir(outputs_dir):
                if comm_dir not in fname and not fname.endswith('.sh'):
                    command = f'rm -rf {os.path.join(outputs_dir, fname)}'
                    print(command)
                    os.system(command)
