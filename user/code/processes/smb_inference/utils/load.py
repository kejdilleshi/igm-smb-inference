import re
import os
import glob
import pandas as pd


def load_inversion_results(filepath):
    """
    Load inversion results from a file and return a list of dictionaries with:
    iter, loss, P, T.
    Skips the last iteration (no loss for it) and assumes loss corresponds to the previous P and T.
    """
    results = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
    file_name = os.path.basename(filepath)

    match = re.search(r'inversion_T(?P<T>[-+]?[0-9]*\.?[0-9]+)_P(?P<P>[-+]?[0-9]*\.?[0-9]+)', file_name)
    if match:
        T = float(match.group('T'))
        P = float(match.group('P'))
    else:
        raise ValueError(f"Filename does not match expected pattern: {file_name}")

    for i in range(len(lines)):
            line = lines[i].strip()

            try:

                iter_part, rest = line.split(':')
                iter_num = int(iter_part.strip().split()[1])
                loss_str, precip_str, T_str = rest.strip().split(',')
                loss = float(loss_str.split('=')[1])
                results.append({'iter': iter_num, 'loss': loss, 'P': P, 'T': T})
                P = float(precip_str.split('=')[1])
                T = float(T_str.split('=')[1])

            except (IndexError, ValueError) as e:
                print(f"Skipping malformed line {i + 1}: {line}")
                continue
    return results


def load_all_inversion_results(directory="."):
    """
    Load all inversion result files matching 'inversion_T*_P*.txt' from the given directory.
    Returns a pandas DataFrame combining all results.
    """
    pattern = os.path.join(directory, "inversion_T*_P*.txt")
    files = glob.glob(pattern)

    all_results = []
    for filepath in files:
        try:
            results = load_inversion_results(filepath)
            all_results.extend(results)
        except Exception as e:
            print(f"Skipping file {filepath} due to error: {e}")
            continue

    return pd.DataFrame(all_results)
