import pandas as pd


def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)
