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

        arg = {k: v for k, v in zip(keys, args)}

        if 'p' in arg['env'] and 'update_weight_interval' in arg and arg['update_weight_interval'] > 1:
            continue

        if arg['arch'] == 'nn':

            # We don't use the replay buffer here.
            if arg['replay'] or arg['buffer_size'] > run_dict['buffer_size'][0]:
                continue

            # Don't include anything that has false for replay buffer and has buffer size larger than first.
            if not arg['replay'] and arg['buffer_size'] > run_dict['buffer_size'][0]:
                continue

            if arg['trunc'] > run_dict['trunc'][0]:
                continue

        elif arg['arch'] == 'lstm':
            if arg['env'] not in ['rg', 'f']:
                continue

        run_string = f"python {main_fname}"

        for k, v in arg.items():

            if v is True:
                run_string += f" --{k}"
            elif v is False:
                continue
            else:
                run_string += f" --{k} {v}"


        run_string += "\n"
        f.write(run_string)
        num_runs += 1

        print(num_runs, run_string)


if __name__ == "__main__":
    runs_dir = Path(ROOT_DIR, 'scripts', 'runs')

    # run_dict is a dictionary with keys as Args keys, and values as lists of parameters you want to run.
    # FOR no-prefilled buffer
    # run_dict = {
    #     'algo': ['sarsa', 'esarsa', 'qlearning'],
    #     'env': ['rg', 'rpg', 'rxg', 'rsg'],
    #     'n_particles': [100],
    #     'batch_size': [64],
    #     'discounting': [0.99],
    #     'p_prefilled': [0.0],
    #     'replay': [True],
    #     'step_size': [0.001, 0.0001, 0.00001],
    #     'buffer_size': [10000, 100000],
    #     'total_steps': [1500000],
    #     'seed': [(i + 2020) for i in range(10)]
    # }

    # run_dict = {
    #     'algo': ['sarsa', 'esarsa', 'qlearning'],
    #     'env': ['rg', 'rxg'],
    #     'n_particles': [100],
    #     'batch_size': [64],
    #     'discounting': [0.99],
    #     'p_prefilled': [0.0],
    #     'replay': [True],
    #     'step_size': [0.001, 0.0001, 0.00001],
    #     'buffer_size': [10000, 100000],
    #     'total_steps': [1500000],
    #     'rock_obs_init': [0.5, 1.],
    #     'seed': [(i + 2020) for i in range(10)]
    # }

    # RockSample LSTM runs
    # run_fname = "runs_rs_lstm.txt"
    # run_fname = "runs_rs_lstm_hidden_update.txt"
    # run_dict = {
    #     'algo': ['sarsa'],
    #     'arch': ['lstm'],
    #     'env': ['rg'],
    #     # 'n_particles': [100],
    #     'batch_size': [64],
    #     'discounting': [0.99],
    #     'p_prefilled': [0.0],
    #     'replay': [True],
    #     'step_size': [0.001, 0.0001, 0.00001],
    #     'trunc': [10, 20],
    #     'buffer_size': [10000, 100000],
    #     'total_steps': [1500000],
    #     # 'er_hidden_update': ['grad'],
    #     'er_hidden_update': ['update'],
    #     'seed': [(i + 2020) for i in range(10)]
    # }

    # Fixed compass world runs
    # run_fname = "run_compass_redo.sh"
    # run_dict = {
    #     'algo': ['sarsa'],
    #     'arch': ['nn'],
    #     'env': ['f', 'fsg', 'fpg'],
    #     'batch_size': [64],
    #     'replay': [False],
    #     'size': [9],
    #     'trunc': [0],
    #     'step_size': [0.001, 0.0001, 0.00001],
    #     'buffer_size': [10000],
    #     'total_steps': [1000000],
    #     'seed': [(i + 2020) for i in range(10)]
    #     # 'seed': [2020]
    # }

    # Fixed compass world LSTM runs
    # run_fname = "runs_compass_lstm.txt"
    run_fname = "runs_compass_lstm_hidden_update.txt"
    run_dict = {
        'algo': ['sarsa'],
        'arch': ['lstm'],
        'env': ['f'],
        'batch_size': [64],
        'replay': [True],
        'trunc': [10, 20],
        'size': [9],
        'step_size': [0.001, 0.0001, 0.00001],
        'buffer_size': [10000, 100000],
        'total_steps': [1000000],
        # 'er_hidden_update': ['grad'],
        'er_hidden_update': ['update'],
        'seed': [(i + 2020) for i in range(10)]
     }

    # Fixed compass world buffer sweep
    # run_fname = "runs_compass_lstm_buffer_sweep.txt"
    # run_dict = {
    #     'algo': ['sarsa'],
    #     'arch': ['lstm'],
    #     'env': ['f'],
    #     'batch_size': [64],
    #     'replay': [True],
    #     'n_hidden': [12],
    #     'trunc': [10],
    #     'size': [9],
    #     'step_size': [0.0001],
    #     'buffer_size': [100, 1000, 5000, 10000, 50000, 100000],
    #     'total_steps': [300000],
    #     'seed': [(i + 2020) for i in range(10)]
    # }

    # Make our runs directory if it doesn't exist
    runs_dir.mkdir(parents=True, exist_ok=True)

    generate_runs(run_dict, runs_dir, runs_fname=run_fname, main_fname='main.py')


