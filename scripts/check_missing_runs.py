import importlib.util
from typing import List
from pathlib import Path
from tqdm import tqdm

from definitions import ROOT_DIR
from unc.utils import load_info


# first get all args from the run file
def isfloat(x):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True


def isint(x):
    try:
        b = int(x)
    except (TypeError, ValueError):
        return False
    else:
        return True


def maybe_convert_value(value):
    if isint(value):
        value = int(value)
    elif isfloat(value):
        value = float(value)
    return value


def compare_args(arg, res_arg):
    for k, v in arg.items():
        if k == 'results_dir':
            continue

        if v != res_arg[k]:
            return False
    return True


if __name__ == "__main__":
    run_file_path = Path(ROOT_DIR, 'scripts', 'runs', 'runs_uf8_cnn_lstm_t10.txt')
    # results_dir_path = Path(ROOT_DIR, 'results')
    results_dir_path = Path(Path.home(), 'scratch', 'uncertainty', 'results')
    all_results_paths = [results_dir_path / 'uf8m_cnn_lstm']

    # run_file_path = Path(ROOT_DIR, 'scripts', 'runs', 'runs_rs_lstm_no_cat.txt')
    # results_dir_path = Path(Path.home(), 'scratch', 'uncertainty', 'results')
    # all_results_paths = [results_dir_path / 'rg_lstm']

    # We first parse all args from the run file
    all_run_args = []
    with open(run_file_path) as f:
        for line in f:
            args = {}
            for k_v_str in line.split('--')[1:]:
                k_v_tuple = k_v_str.split(' ')
                key = k_v_tuple[0]

                if len(k_v_tuple) == 3:
                    # we have --key value
                    value = k_v_tuple[1]
                    args[key] = maybe_convert_value(value)

                elif len(k_v_tuple) == 2:
                    if not k_v_tuple[-1]:
                        args[key] = True
                    else:
                        # we have --key value, and we need to strip value of newlines
                        value = k_v_tuple[1].replace('\n', '')

                        args[key] = maybe_convert_value(value)
                else:
                    raise NotImplementedError
            all_run_args.append(args)

    # now we get the args from the results
    all_res_args = []
    for res_path in all_results_paths:

        for res in res_path.iterdir():
            if res.is_file() and res.suffix == ".npy":
                info = load_info(res)
                args = info['args'].item()
                all_res_args.append(args)

    need_to_run = []

    # now we do our matching. This may take a while.
    # order in this list corresponds to line number.
    for line_num, arg in tqdm(enumerate(all_run_args)):
        found_match = False

        for res_arg in all_res_args:
            if compare_args(arg, res_arg):
                found_match = True
                break

        if not found_match:
            need_to_run.append(line_num + 1)

    print(f"Line numbers needed to re-run: {','.join(str(n) for n in need_to_run)}")



