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
        'env': ['sr', 'r'],
        'seed': [(i + 2020) for i in range(30)]
    }

    runs_path = runs_dir / runs_fname

    if runs_path.is_file():
        runs_path.unlink()

    f = open(runs_path, 'a+')

    keys, values = [], []
    for k, v in run_dict.items():
        keys.append(k)
        values.append(v)

    for i, args in enumerate(product(*values)):
        run_string = "python main.py"
        for k, v in zip(keys, args):
            run_string += f" --{k.replace('_', '-')} {v}"
        run_string += "\n"
        f.write(run_string)

        print(i, run_string)


if __name__ == "__main__":
    runs_dir = Path(ROOT_DIR, 'scripts', 'runs')

    # Make our runs directory if it doesn't exist
    runs_dir.mkdir(parents=True, exist_ok=True)

    generate_runs(runs_dir)


