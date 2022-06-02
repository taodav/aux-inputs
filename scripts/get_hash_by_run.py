from pathlib import Path
from tqdm import tqdm

from definitions import ROOT_DIR
from unc.args import Args, hash_training_args
from unc.utils import load_info


if __name__ == "__main__":
    line_nums = [
        66,67
    ]
    run_file_path = Path(ROOT_DIR, 'scripts', 'runs', 'runs_uf8_cnn_lstm_t20.txt')

    # results_dir_path = Path(Path.home(), 'scratch', 'uncertainty', 'results')

    # We parse all args from the run file if the line numbers match
    all_run_args = []
    with open(run_file_path) as f:
        for i, line in enumerate(f):
            if i + 1 in line_nums:
                parser = Args()
                arg_strings = line.split(' ')[2:]
                args = parser.parse_args(arg_strings)

                arg_hash = hash_training_args(args)
                print(f"File: {run_file_path}, line: {i}, hash: {arg_hash}")









