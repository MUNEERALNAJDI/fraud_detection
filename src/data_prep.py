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

def clean_cdr(df, drop_cols=None, verbose=True):
    """
    Basic cleaning for CDR data:
    - Drop rows with empty phone_no_m
    - Keep only valid call durations (>= 0)
    - Drop specified columns
    Prints before/after status if verbose=True.
    """
    
    if verbose:
        print("🔹 Initial shape:", df.shape)
    
    # Drop empty phone numbers
    if "phone_no_m" in df.columns:
        before = df.shape[0]
        df = df.dropna(subset=["phone_no_m"])
        if verbose:
            print(f"✅ Dropped empty phone_no_m: {before - df.shape[0]} rows removed, shape now {df.shape}")

    # Keep only valid call durations (>=0)
    if "call_dur" in df.columns:
        before = df.shape[0]
        df = df[df["call_dur"] >= 0]
        if verbose:
            print(f"✅ Filtered invalid call_dur (<0): {before - df.shape[0]} rows removed, shape now {df.shape}")

    # Drop specified columns
    if drop_cols:
        before = set(df.columns)
        df = df.drop(columns=drop_cols, errors="ignore")
        after = set(df.columns)
        dropped = before - after
        if verbose:
            print(f"✅ Dropped columns: {list(dropped)} | Remaining cols: {list(after)}")
    
    if verbose:
        print("🔹 Final shape:", df.shape)

    return df


