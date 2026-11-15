"""Shared data loading and the fixed day-based split used by every model/experiment."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Occupancy_Estimation.csv"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

TARGET = "Room_Occupancy_Count"
LABELS = [0, 1, 2, 3]

# Fixed day-based split (same for ALL models and experiments).
# Only 3 of the 7 days contain occupied samples (12/22, 12/23, 01/10), and class 1
# appears only on 12/22 and 12/23. Each split therefore gets exactly one occupied day,
# so no occupancy session is shared between splits. Train and test both contain all
# 4 classes; validation (01/10) has no class-1 samples.
SPLIT_DAYS = {
    "train": ["2017/12/22", "2017/12/24", "2017/12/25", "2017/12/26"],
    "val": ["2018/01/10", "2018/01/11"],
    "test": ["2017/12/23"],
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    return df.sort_values("Timestamp").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    # Date/Time are identifiers, not sensor readings; they are excluded to avoid leakage.
    return [c for c in df.columns if c not in ("Date", "Time", "Timestamp", TARGET)]


def get_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    days = [d for v in SPLIT_DAYS.values() for d in v]
    assert len(days) == len(set(days)), "a day is used in more than one split"
    assert set(days) == set(df["Date"].unique()), "every day must be assigned exactly once"
    return {name: df[df["Date"].isin(ds)].reset_index(drop=True) for name, ds in SPLIT_DAYS.items()}
