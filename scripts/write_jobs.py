from typing import List
from pathlib import Path
from itertools import product
from definitions import ROOT_DIR


def generate_runs(runs_dir: Path, runs_fname: str = 'runs.txt') -> List[str]:
    """
    Generate strings for each run.
    :return:
    """
    # run_dict is a dictionary with keys as Args keys, and values as lists of parameters you want to run.
    run_dict = {
        'env': ['f', 'fs', 'fp', 'fpm', 'fpv'],
        # 'env': ['fpv'],
        'update_weight_interval': [1],
        'seed': [(i + 2020) for i in range(30)],
        'total_steps': [150000]
    }
    # run_dict = {
    #     'env': ['pv'],
    #     'update_weight_interval': [1],
    #     'seed': [(i + 2020) for i in range(30)]
    # }

    runs_path = runs_dir / runs_fname

    if runs_path.is_file():
        runs_path.unlink()

    f = open(runs_path, 'a+')

    keys, values = [], []
    for k, v in run_dict.items():
        keys.append(k)
        values.append(v)
    num_runs = 0
    for i, args in enumerate(product(*values)):
        run_string = "python main.py"

        # We do some filtering based on whether or not this is a particle filter run
        is_pf_run = False
        for k, v in zip(keys, args):
            if k == "env" and "p" in v:
                is_pf_run = True

        skip = False
        for k, v in zip(keys, args):
            if not is_pf_run and k == 'update_weight_interval' and v > 1:
                # Since non-pf runs don't use this hyperparam, skip the ones not needed
                skip = True
                break
            run_string += f" --{k} {v}"

        if skip:
            continue

        run_string += "\n"
        f.write(run_string)
        num_runs += 1

        print(num_runs, run_string)


if __name__ == "__main__":
    runs_dir = Path(ROOT_DIR, 'scripts', 'runs')

    # Make our runs directory if it doesn't exist
    runs_dir.mkdir(parents=True, exist_ok=True)

    generate_runs(runs_dir)


