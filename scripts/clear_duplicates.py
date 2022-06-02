from time import strptime
from pathlib import Path

from definitions import ROOT_DIR


if __name__ == "__main__":
    results_dir_path = Path(Path.home(), 'scratch', 'uncertainty', 'results')
    # results_dir_path = Path(ROOT_DIR, 'results')
    all_results_paths = [results_dir_path / 'uf8a_cnn']

    unique_hashes = {}

    for res_path in all_results_paths:
        for res in res_path.iterdir():
            if res.is_file() and res.suffix == ".npy":
                arg_hash, date_suffix = res.name.split('_')
                date_str = date_suffix.split('.')[0]
                date_obj = strptime(date_str, "%a %b %d %H:%M:%S %Y")

                if arg_hash not in unique_hashes:
                    unique_hashes[arg_hash] = []

                unique_hashes[arg_hash].append((date_obj, res))

    for hash, results in unique_hashes.items():
        sorted_results = sorted(results, key=lambda x: x[0])
        for date_obj, res in sorted_results[:-1]:
            print(f"Deleting file {res}")
            res.unlink()

