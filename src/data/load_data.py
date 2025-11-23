# src/data/load_data.py

import pandas as pd
from pathlib import Path

class IoTDatasetLoader:
    def __init__(self, base_path):
        self.base_path = Path(base_path)

    def list_files(self):
        return sorted([str(p) for p in self.base_path.rglob("*.csv")])

    def load_single(self, filepath, nrows=None):
        return pd.read_csv(filepath, nrows=nrows)

    def load_all(self, sample=False, nrows=50000):
        files = self.list_files()
        dfs = []

        for f in files:
            if sample:
                df = pd.read_csv(f, nrows=nrows)
            else:
                df = pd.read_csv(f)
            df["source_file"] = Path(f).name
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)