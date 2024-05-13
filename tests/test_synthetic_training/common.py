import csv
import numpy as np


def load_csv(file_path:str) -> np.ndarray:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return np.array(data, dtype=float)

