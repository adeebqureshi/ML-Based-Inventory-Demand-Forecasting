import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

MODEL_FEATURE_NAMES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_7",
    "rolling_std_7",
    "trend",
    "day_of_week",
    "month",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Lag features
    df["lag_1"] = df["quantity_sold"].shift(1)
    df["lag_2"] = df["quantity_sold"].shift(2)
    df["lag_3"] = df["quantity_sold"].shift(3)

    # Rolling statistics
    df["rolling_mean_7"] = df["quantity_sold"].rolling(7).mean()
    df["rolling_std_7"] = df["quantity_sold"].rolling(7).std()

    df["trend"] = np.arange(len(df))

    df = df.dropna()

    return df


def train_model(df: pd.DataFrame, split_ratio: float = 0.8):

    df = create_features(df)

    X = df[MODEL_FEATURE_NAMES]
    y = df["quantity_sold"]

    split = int(len(df) * split_ratio)

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_test = X.iloc[split:]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)

    metrics = {
        "MAPE": round(mape * 100, 2)
    }

    return model, split, y, y_pred, metrics


def forecast_next_7(df, model):

    df_feat = create_features(df)

    last_row = df_feat.iloc[-1:].copy()

    forecasts = []
    dates = []

    current_date = df["date"].max()

    for i in range(7):

        current_date = current_date + timedelta(days=1)

        last_row["day_of_week"] = current_date.dayofweek
        last_row["month"] = current_date.month

        pred = model.predict(last_row[MODEL_FEATURE_NAMES])[0]

        pred = max(0, pred)

        forecasts.append(round(pred, 2))
        dates.append(current_date)

        last_row["lag_3"] = last_row["lag_2"]
        last_row["lag_2"] = last_row["lag_1"]
        last_row["lag_1"] = pred

    return pd.DataFrame({
        "date": dates,
        "forecast": forecasts
    })