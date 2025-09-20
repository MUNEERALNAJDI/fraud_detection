import os
import pandas as pd
import yaml
import logging

# ==========================
# Logging Setup
# ==========================


# Make sure results directory exists
os.makedirs("results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/data_prep.log")
    ]
)
logger = logging.getLogger(__name__)



# ==========================
# Config Loader
# ==========================
def load_config(config_path="configs/baseline.yaml"):
    """Load YAML config file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


# ==========================
# Data Loaders
# ==========================
def load_cdr(file_path, nrows=None):
    """Load a CDR CSV file with optional row limit (sample mode)."""
    logger.info(f"Loading file: {file_path} with nrows={nrows}")
    return pd.read_csv(file_path, nrows=nrows)


def load_all_data(config):
    """
    Load all CSVs defined in config into a dict of DataFrames.
    Uses training.sample_size if available.
    """
    base = config["dataset"]["base_path"]
    files = config["dataset"]["files"]
    sample_size = config.get("training", {}).get("sample_size", None)

    data = {}
    for name, fname in files.items():
        path = os.path.join(base, fname)
        df = load_cdr(path, nrows=sample_size)
        data[name] = df
        logger.info(f"Loaded {name} -> {df.shape} from {path}")
    return data


# ==========================
# Cleaning Functions
# ==========================
def clean_cdr(df, drop_cols=None, verbose=True):
    """
    Basic cleaning for CDR data:
    - Drop rows with empty phone_no_m
    - Keep only valid call durations (>= 0)
    - Drop specified columns
    """

    if verbose:
        logger.info(f"Initial shape: {df.shape}")

    # Drop empty phone numbers
    if "phone_no_m" in df.columns:
        before = df.shape[0]
        df = df.dropna(subset=["phone_no_m"])
        logger.info(f"Dropped empty phone_no_m: {before - df.shape[0]} rows")

    # Keep only valid call durations
    if "call_dur" in df.columns:
        before = df.shape[0]
        df = df[df["call_dur"] >= 0]
        logger.info(f"Filtered invalid call_dur: {before - df.shape[0]} rows")

    # Drop specified columns
    if drop_cols:
        before = set(df.columns)
        df = df.drop(columns=drop_cols, errors="ignore")
        after = set(df.columns)
        dropped = before - after
        logger.info(f"Dropped columns: {list(dropped)}")

    logger.info(f"Final shape: {df.shape}")
    return df


# ==========================
# EDA Helper
# ==========================
def quick_eda(df, target_col=None):
    """Quick summary: shape, NA counts, fraud ratio (if target_col given)."""
    summary = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_values": int(df.isnull().sum().sum())
    }
    if target_col and target_col in df.columns:
        ratio = df[target_col].value_counts(normalize=True).to_dict()
        summary["fraud_ratio"] = ratio
    logger.info(f"EDA summary: {summary}")
    return summary
