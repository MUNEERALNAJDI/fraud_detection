import pandas as pd
import os
import yaml

def load_config(config_path="configs/baseline.yaml"):
    """Load YAML config file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_cdr(file_path, nrows=None):
    """Load a CDR CSV file with optional row limit (sample mode)."""
    return pd.read_csv(file_path, nrows=nrows)

def load_all_data(config, nrows=None):
    """Load all CSVs defined in config into a dict of DataFrames."""
    base = config["dataset"]["base_path"]
    files = config["dataset"]["files"]
    data = {}
    for name, fname in files.items():
        path = os.path.join(base, fname)
        data[name] = load_cdr(path, nrows=nrows)
    return data

def clean_cdr(df, drop_cols=None):
    """Basic cleaning: drop empty IDs, remove invalid durations, drop columns."""
    if "phone_no_m" in df.columns:
        df = df.dropna(subset=["phone_no_m"])
    if "call_dur" in df.columns:
        df = df[df["call_dur"] > 0]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df

