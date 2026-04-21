import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from typing import Tuple, Any

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

FEATURES = ["lag_1", "lag_2", "lag_3", "rolling_mean_7", "rolling_std_7", "trend"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series features required for forecasting.
    Shared by both RF and XGBoost pipelines — single source of truth.
    """
    df = df.sort_values("date").copy()
    df["lag_1"]          = df["quantity_sold"].shift(1)
    df["lag_2"]          = df["quantity_sold"].shift(2)
    df["lag_3"]          = df["quantity_sold"].shift(3)
    df["rolling_mean_7"] = df["quantity_sold"].rolling(7).mean()
    df["rolling_std_7"]  = df["quantity_sold"].rolling(7).std()
    df["trend"]          = np.arange(len(df))
    df = df.dropna()
    return df


def _calc_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return round(float(mape * 100), 2)


def _recursive_forecast_7(df_feat: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Shared 7-day recursive forecasting logic.
    Works with any sklearn-compatible model (RF or XGBoost).
    """
    last_date     = df_feat["date"].max()
    prev1         = float(df_feat.iloc[-1]["quantity_sold"])
    prev2         = float(df_feat.iloc[-2]["quantity_sold"])
    prev3         = float(df_feat.iloc[-3]["quantity_sold"])
    recent_window = list(df_feat["quantity_sold"].tail(7).astype(float))
    rolling_mean  = float(np.mean(recent_window))
    rolling_std   = float(np.std(recent_window, ddof=1)) if len(recent_window) > 1 else 0.0
    trend         = float(df_feat["trend"].iloc[-1])

    dates, preds = [], []
    for i in range(1, 8):
        feat_row = pd.DataFrame({
            "lag_1":          [prev1],
            "lag_2":          [prev2],
            "lag_3":          [prev3],
            "rolling_mean_7": [rolling_mean],
            "rolling_std_7":  [rolling_std],
            "trend":          [trend + i],
        })
        prediction = max(0.0, float(model.predict(feat_row)[0]))
        dates.append(last_date + timedelta(days=i))
        preds.append(round(prediction, 2))

        prev3 = prev2
        prev2 = prev1
        prev1 = prediction
        recent_window.append(prediction)
        recent_window.pop(0)
        rolling_mean = float(np.mean(recent_window))
        rolling_std  = float(np.std(recent_window, ddof=1)) if len(recent_window) > 1 else 0.0

    return pd.DataFrame({"date": dates, "predicted_demand": preds})


def train_model(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple:
    """
    Train Random Forest model.

    Returns
    -------
    Tuple: (model, split, y, y_pred, metrics)
        model   : trained RandomForestRegressor
        split   : integer index where train/test split occurs
        y       : full target Series (all rows after feature dropna)
        y_pred  : numpy array of predictions on the test slice
        metrics : dict with key "MAPE" (float, already multiplied by 100)
    """
    df_feat = create_features(df)
    
    # Validate minimum dataset size
    if len(df_feat) < 10:
        raise ValueError(f"Insufficient data: need at least 10 records, got {len(df_feat)}")
    
    X       = df_feat[FEATURES]
    y       = df_feat["quantity_sold"]
    split   = int(len(df_feat) * split_ratio)
    
    # Validate split produces both train and test sets
    if split < 2 or split >= len(df_feat) - 1:
        raise ValueError(f"Invalid train/test split: {split} out of {len(df_feat)} records")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
    )
    model.fit(X.iloc[:split], y.iloc[:split])
    y_pred  = model.predict(X.iloc[split:])
    
    # Validate predictions
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("Model produced invalid predictions (NaN or Inf values)")
    
    metrics = {"MAPE": _calc_mape(y.iloc[split:], y_pred)}

    return model, split, y, y_pred, metrics


def train_xgboost(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple:
    """
    Train XGBoost model on identical feature set as Random Forest.

    Returns
    -------
    Tuple: (model, split, y, y_pred, metrics)
        Returns (None, 0, None, None, {}) if XGBoost is not installed.
    """
    if not XGBOOST_AVAILABLE:
        return None, 0, None, None, {}

    df_feat = create_features(df)
    
    # Validate minimum dataset size
    if len(df_feat) < 10:
        raise ValueError(f"Insufficient data: need at least 10 records, got {len(df_feat)}")
    
    X       = df_feat[FEATURES]
    y       = df_feat["quantity_sold"]
    split   = int(len(df_feat) * split_ratio)
    
    # Validate split produces both train and test sets
    if split < 2 or split >= len(df_feat) - 1:
        raise ValueError(f"Invalid train/test split: {split} out of {len(df_feat)} records")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    
    try:
        model.fit(
            X.iloc[:split], y.iloc[:split],
            eval_set=[(X.iloc[split:], y.iloc[split:])],
            verbose=False,
        )
    except Exception as e:
        raise ValueError(f"XGBoost training failed: {str(e)}")
    
    y_pred  = model.predict(X.iloc[split:])
    
    # Validate predictions
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("XGBoost produced invalid predictions (NaN or Inf values)")
    
    metrics = {"MAPE": _calc_mape(y.iloc[split:], y_pred)}

    return model, split, y, y_pred, metrics


def forecast_next_7(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    """
    Forecast demand for the next 7 days using recursive single-step prediction.
    Compatible with both RandomForestRegressor and XGBRegressor.
    """
    df_feat = create_features(df)
    return _recursive_forecast_7(df_feat, model)