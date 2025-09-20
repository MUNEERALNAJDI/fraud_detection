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

# ==========================
# Feature Table Builder
# ==========================
def build_feature_table(data: dict):
    """
    Merge APP, SMS, USER, VOC datasets on phone_no_m into a single feature table.
    Each dataset is aggregated per phone_no_m, then merged into one DataFrame.
    
    Parameters
    ----------
    data : dict
        Dictionary of raw DataFrames, e.g. {"app": df_app, "sms": df_sms, "user": df_user, "voc": df_voc}
    
    Returns
    -------
    feature_df : pd.DataFrame
        Merged feature table with one row per phone_no_m
    """

    feature_parts = []

    # --------------------------
    # APP features
    # --------------------------
    if "app" in data:
        df = data["app"].copy()
        app_feat = df.groupby("phone_no_m").agg(
            app_count=("busi_name", "nunique"),
            total_flow=("flow", "sum"),
            avg_flow=("flow", "mean")
        ).reset_index()
        feature_parts.append(app_feat)

    # --------------------------
    # SMS features
    # --------------------------
    if "sms" in data:
        df = data["sms"].copy()
        sms_feat = df.groupby("phone_no_m").agg(
            sms_count=("opposite_no_m", "count"),
            unique_contacts=("opposite_no_m", "nunique")
        ).reset_index()
        feature_parts.append(sms_feat)

    # --------------------------
    # USER features (static profile)
    # --------------------------
    if "user" in data:
        df = data["user"].copy()
        user_feat = df.drop_duplicates(subset=["phone_no_m"])
        feature_parts.append(user_feat)

    # --------------------------
    # VOC features (calls)
    # --------------------------
    if "voc" in data:
        df = data["voc"].copy()
        voc_feat = df.groupby("phone_no_m").agg(
            call_count=("opposite_no_m", "count"),
            unique_callers=("opposite_no_m", "nunique"),
            avg_call_dur=("call_dur", "mean"),
            total_call_dur=("call_dur", "sum")
        ).reset_index()
        feature_parts.append(voc_feat)

    # --------------------------
    # Merge all features
    # --------------------------
    from functools import reduce
    feature_df = reduce(
        lambda left, right: pd.merge(left, right, on="phone_no_m", how="outer"),
        feature_parts
    )

    # --------------------------
    # Handle missing values
    # --------------------------
    feature_df = feature_df.fillna(0)

    return feature_df

from ydata_profiling import ProfileReport

profile = ProfileReport(
    feature_df,
    title="Fraud Detection EDA Report",
    explorative=True,
    correlations={"cramers": {"calculate": False}},   # optional speed-up
    missing_diagrams={"heatmap": False},              # disable some heavy plots
    interactions={"continuous": False},               # skip unnecessary
    samples=None                                      # skip samples
)

profile.to_file("results/feature_profile.html")



from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_features(df, numeric_cols, categorical_cols):
    """
    Apply scaling to numeric cols and one-hot encode categorical cols.
    Returns transformed DataFrame ready for ML.
    """
    # Scale numeric
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode categorical
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Merge back
    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)

    return df
