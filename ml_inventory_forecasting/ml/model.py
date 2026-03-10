import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from typing import Tuple


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series features required for forecasting.
    """

    df = df.sort_values("date")

    df["lag_1"] = df["quantity_sold"].shift(1)
    df["lag_2"] = df["quantity_sold"].shift(2)
    df["lag_3"] = df["quantity_sold"].shift(3)

    df["rolling_mean_7"] = df["quantity_sold"].rolling(7).mean()
    df["rolling_std_7"] = df["quantity_sold"].rolling(7).std()

    df["trend"] = np.arange(len(df))

    df = df.dropna()

    return df


def train_model(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple:
    """
    Train Random Forest model and return model, split index, actuals,
    predictions, and a metrics dict containing MAPE.

    Returns
    -------
    Tuple: (model, split, y, y_pred, metrics)
        model     : trained RandomForestRegressor
        split     : integer index where train/test split occurs
        y         : full target Series (all rows after feature dropna)
        y_pred    : numpy array of predictions on the test slice
        metrics   : dict with key "MAPE" (float, already multiplied by 100)
    """

    df = create_features(df)

    features = [
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_7",
        "rolling_std_7",
        "trend",
    ]

    X = df[features]
    y = df["quantity_sold"]

    split = int(len(df) * split_ratio)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X.iloc[:split], y.iloc[:split])

    y_pred = model.predict(X.iloc[split:])

    # evaluation metric
    mape = mean_absolute_percentage_error(y.iloc[split:], y_pred)

    metrics = {
        "MAPE": round(mape * 100, 2)
    }

    return model, split, y, y_pred, metrics


def forecast_next_7(df: pd.DataFrame, model: RandomForestRegressor) -> pd.DataFrame:
    """
    Forecast demand for the next 7 days using recursive single-step prediction.
    """

    df = create_features(df)

    last_date = df["date"].max()

    prev1 = df.iloc[-1]["quantity_sold"]
    prev2 = df.iloc[-2]["quantity_sold"]
    prev3 = df.iloc[-3]["quantity_sold"]

    # Seed rolling stats from the last 7 observed values
    recent_window = list(df["quantity_sold"].tail(7))
    rolling_mean = float(np.mean(recent_window))
    rolling_std = float(np.std(recent_window, ddof=1)) if len(recent_window) > 1 else 0.0

    trend = df["trend"].iloc[-1]

    dates = []
    preds = []

    for i in range(1, 8):

        next_date = last_date + timedelta(days=i)

        features = pd.DataFrame({
            "lag_1": [prev1],
            "lag_2": [prev2],
            "lag_3": [prev3],
            "rolling_mean_7": [rolling_mean],
            "rolling_std_7": [rolling_std],
            "trend": [trend + i],
        })

        prediction = model.predict(features)[0]
        prediction = max(0, prediction)

        dates.append(next_date)
        preds.append(round(float(prediction), 2))

        # Update lag variables with the new prediction
        prev3 = prev2
        prev2 = prev1
        prev1 = prediction

        # FIX: update both rolling_mean AND rolling_std each step
        # so the uncertainty estimate stays current across the 7-day horizon.
        recent_window.append(prediction)
        recent_window.pop(0)  # keep window at 7 elements
        rolling_mean = float(np.mean(recent_window))
        rolling_std = float(np.std(recent_window, ddof=1)) if len(recent_window) > 1 else 0.0

    forecast_df = pd.DataFrame({
        "date": dates,
        "predicted_demand": preds,
    })

    return forecast_df