import argparse
import importlib.util
from typing import List
from pathlib import Path
from itertools import product

from definitions import ROOT_DIR


def generate_runs(run_dicts: List[dict], runs_dir: Path, runs_fname: str = 'runs.txt',
                  main_fname: str = 'main.py',
                  results_dir: Path = None,
                  log_dir: Path = None) -> List[str]:
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

    num_runs = 0
    for run_dict in run_dicts:
        keys, values = [], []
        for k, v in run_dict.items():
            keys.append(k)
            values.append(v)
        for i, args in enumerate(product(*values)):

            arg = {k: v for k, v in zip(keys, args)}

            if 'p' in arg['env'] and 'update_weight_interval' in arg and arg['update_weight_interval'] > 1:
                continue

            # we don't have uncertainty decay for uf{}a
            if 'uf' in arg['env'] and 'a' in arg['env']:
                if arg['uncertainty_decay'] < 1.:
                    continue
                if 'distance_noise' in arg and 'distance_unc_encoding' in arg:
                    if arg['distance_noise'] or arg['distance_unc_encoding']:
                        continue

            # if distance_noise == False, then distance_unc_encoding can't be true
            if 'distance_noise' in arg and 'distance_unc_encoding' in arg:
                if 'uf' in arg['env'] and not arg['distance_noise'] and arg['distance_unc_encoding']:
                    continue

            run_string = f"python {main_fname}"

            for k, v in arg.items():

                if v is True:
                    run_string += f" --{k}"
                elif v is False or v is None:
                    continue
                else:
                    run_string += f" --{k} {v}"

            if results_dir is not None:
                run_string += f" --results_dir {results_dir}"

            if log_dir is not None:
                run_string += f" --log_dir {log_dir}"

            run_string += "\n"
            f.write(run_string)
            num_runs += 1

            print(num_runs, run_string)


def import_module_to_hparam(hparam_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("hparam", hparam_path)
    hparam_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hparam_module)
    hparams = hparam_module.hparams
    return hparams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparam', default='', type=str)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()

    runs_dir = Path(ROOT_DIR, 'scripts', 'runs')

    hparam_path = Path(ROOT_DIR, 'scripts', 'hparams', args.hparam + ".py")
    hparams = import_module_to_hparam(hparam_path)

    results_dir, log_dir = None, None
    if not args.local:
        # Here we assume we want to write to the scratch directory in CC.
        results_dir = Path("/home/taodav/scratch/uncertainty/results")
        log_dir = Path("/home/taodav/scratch/uncertainty/log")

    # Make our directories if it doesn't exist
    runs_dir.mkdir(parents=True, exist_ok=True)

    generate_runs(hparams['args'], runs_dir, runs_fname=hparams['file_name'], main_fname='main.py',
                  results_dir=results_dir, log_dir=log_dir)


