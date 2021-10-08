from typing import List
from pathlib import Path
from itertools import product
from definitions import ROOT_DIR


def generate_runs(run_dict: dict, runs_dir: Path, runs_fname: str = 'runs.txt', main_fname: str = 'main.py') -> List[str]:
    """
    :param runs_dir: Directory to put the runs
    :param runs_fname: What do we call our run file?
    :param main_fname: what is our python entry script?
    :return:
    """

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
        run_string = f"python {main_fname}"

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
            if v is True:
                run_string += f" --{k}"
            elif v is False:
                continue
            else:
                run_string += f" --{k} {v}"

        if skip:
            continue

        run_string += "\n"
        f.write(run_string)
        num_runs += 1

        print(num_runs, run_string)


if __name__ == "__main__":
    runs_dir = Path(ROOT_DIR, 'scripts', 'runs')

    # run_dict is a dictionary with keys as Args keys, and values as lists of parameters you want to run.
    # FOR no-prefilled buffer
    run_dict = {
        'algo': ['sarsa', 'esarsa', 'qlearning'],
        'env': ['rg', 'rpg', 'rxg', 'rsg'],
        'n_particles': [100],
        'seed': [(i + 2020) for i in range(10)],
        'batch_size': [64],
        'discounting': [0.99],
        'p_prefilled': [0.0],
        'step_size': [0.001, 0.0001, 0.00001],
        'buffer_size': [10000, 20000, 100000],
        'total_steps': [int(1e6)]
    }

    # run_dict = {
    #     'algo': ['sarsa', 'esarsa', 'qlearning'],
    #     'env': ['rg', 'rpg', 'rxg', 'rsg'],
    #     'n_particles': [100],
    #     'seed': [(i + 2020) for i in range(10)],
    #     'batch_size': [64],
    #     'discounting': [0.99],
    #     'p_prefilled': [0.25],
    #     'buffer_size': [10000, 20000, 100000],
    #     'total_steps': [int(5e5)]
    # }

    # Make our runs directory if it doesn't exist
    runs_dir.mkdir(parents=True, exist_ok=True)

    generate_runs(run_dict, runs_dir, runs_fname='runs_rs_prefilled.txt', main_fname='scripts/run_double_buffer.py')


