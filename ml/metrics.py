import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    r2 = r2_score(y_true, y_pred)

    # Protect against division by zero
    y_true_safe = np.where(y_true == 0, 1e-8, y_true)

    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": round(float(mape), 2)
    }