import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names in the dataset.

    Converts all column names to lowercase, removes leading/trailing spaces,
    and replaces spaces with underscores.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with normalized column names.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df


def detect_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], Optional[str]]:
    """
    Automatically detect important columns from the dataset.

    This function attempts to identify:
    - Date columns
    - Quantity sold columns
    - Product identifier column
    """

    date_cols = [c for c in df.columns if "date" in c or "day" in c]

    qty_cols = [c for c in df.columns if "qty" in c or "sold" in c or "quantity" in c]

    product_col = next(
        (
            c for c in df.columns
            if any(k in c for k in ["product", "item", "item_name", "product_id", "sku"])
        ),
        None,
    )

    return date_cols, qty_cols, product_col


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean the dataframe before model training.

    This function handles date parsing, sorting, and basic calendar
    features. The lag/rolling features used by the model are created
    inside model.py's create_features() to keep a single source of
    truth for the feature set.

    FIX: previously this function created lag_1, lag_3, lag_7 which did
    NOT match the model's expected features (lag_1, lag_2, lag_3). The
    mismatched columns were silently unused. Now prepare_features() only
    adds calendar columns; model.py owns the lag/rolling feature logic.
    """

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.sort_values("date").dropna(subset=["date", "quantity_sold"])

    # Calendar features (informational / used by future extensions)
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    return df