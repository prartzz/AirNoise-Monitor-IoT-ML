# train_multi.py - enhanced
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def load_custom_holidays():
    """Load custom holidays from CSV into a set of dates."""
    holiday_set = set()
    custom_holidays_path = os.path.join("dataset", "custom_holidays.csv")
    if os.path.exists(custom_holidays_path):
        try:
            custom_df = pd.read_csv(custom_holidays_path, parse_dates=["date"])
            custom_set = set(custom_df["date"].dt.date)
            holiday_set.update(custom_set)
        except Exception as e:
            print(f"Warning: Could not load custom holidays: {e}")
    return holiday_set

# ================= CONFIG =================
SYNTHETIC_PATH = "dataset/synthetic_data.csv"
CLEANED_TS_PATH = "dataset/cleaned_ts_data.csv"
MERGED_PATH = "dataset/merged_train_data.csv"
MODEL_OUT_DIR = "models"
MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "multi_model.joblib")
RANDOM_STATE = 42

os.makedirs(MODEL_OUT_DIR, exist_ok=True)


def load_and_merge():
    # Prefer merged file if already created by fetch_preprocess
    if os.path.exists(MERGED_PATH) and os.path.getsize(MERGED_PATH) > 0:
        df = pd.read_csv(MERGED_PATH, parse_dates=["Timestamp"])
    else:
        # Read synthetic and cleaned TS if available
        parts = []
        if os.path.exists(SYNTHETIC_PATH) and os.path.getsize(SYNTHETIC_PATH) > 0:
            parts.append(pd.read_csv(SYNTHETIC_PATH, parse_dates=["Timestamp"]))
        if os.path.exists(CLEANED_TS_PATH) and os.path.getsize(CLEANED_TS_PATH) > 0:
            try:
                parts.append(pd.read_csv(CLEANED_TS_PATH, parse_dates=["Timestamp"]))
            except Exception:
                pass
        if parts:
            df = pd.concat(parts, ignore_index=True)
            # If an old dataset column `Is_Hostel_Near_Road` is present, drop it.
            if "Is_Hostel_Near_Road" in df.columns:
                df = df.drop(columns=["Is_Hostel_Near_Road"])
            # Coerce to datetime (handles mixed types), drop bad rows
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp"]).reset_index(drop=True)
            # If dtype is timezone-aware, convert to naive UTC timestamps
            try:
                from pandas.api import types as ptypes
                if ptypes.is_datetime64tz_dtype(df["Timestamp"].dtype):
                    df["Timestamp"] = df["Timestamp"].dt.tz_convert(None)
            except Exception:
                # older pandas or unexpected state; ignore
                pass
            df = df.sort_values("Timestamp").reset_index(drop=True)
        else:
            raise FileNotFoundError("No training data found. Put synthetic_data.csv or run fetch_preprocess.py")
    return df


def ensure_context_features(df: pd.DataFrame, custom_holidays=None) -> pd.DataFrame:
    df = df.copy()
    if "Day" not in df.columns and "Timestamp" in df.columns:
        df["Day"] = df["Timestamp"].dt.day_name()
    if "Hour" not in df.columns and "Timestamp" in df.columns:
        df["Hour"] = df["Timestamp"].dt.hour
    if "Day_num" not in df.columns and "Timestamp" in df.columns:
        df["Day_num"] = df["Timestamp"].dt.weekday
    if "Is_Holiday" not in df.columns:
        if custom_holidays is None:
            custom_holidays = load_custom_holidays()
        df["Is_Holiday"] = df["Timestamp"].dt.date.apply(lambda d: d in custom_holidays)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Prev_PM2_5"] = df["PM2_5"].shift(1)
    df["Prev_PM10"] = df["PM10"].shift(1)
    df["Prev_Noise"] = df["Noise"].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df


def train_and_save_model(df: pd.DataFrame):
    custom_holidays = load_custom_holidays()
    df = ensure_context_features(df, custom_holidays=custom_holidays)
    df = add_lag_features(df)

    FEATURES = [
        "Day_num",
        "Hour",
        "Prev_PM2_5",
        "Prev_PM10",
        "Prev_Noise",
        "Is_Holiday",
    ]
    TARGETS = ["PM2_5", "PM10", "Noise"]

    X = df[FEATURES]
    y = df[TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_OUT_PATH)
    print(f"Saved model to {MODEL_OUT_PATH}")

    # Evaluate
    y_pred = model.predict(X_test)
    mae_per_target = {}
    for i, t in enumerate(TARGETS):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        mae_per_target[t] = mae

    print("\nMean Absolute Error per target:")
    for t, v in mae_per_target.items():
        print(f" - {t}: {v:.3f}")

    # Save predictions alongside timestamps for inspection
    test_start_idx = len(X_train)
    out_df = df.copy()
    pred_df = pd.DataFrame(np.nan, index=out_df.index, columns=[f"Pred_{t}" for t in TARGETS])
    pred_df.iloc[test_start_idx:, :] = y_pred
    out_df = pd.concat([out_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    out_df.to_csv("dataset/predictions_with_timestamps.csv", index=False)
    print("Saved predictions to dataset/predictions_with_timestamps.csv")


def predict_for_datetime(dt: pd.Timestamp, df_history: pd.DataFrame, model=None):
    # Ensure df_history has been prepared (no NaNs and contains necessary cols)
    custom_holidays = load_custom_holidays()
    df_h = ensure_context_features(df_history, custom_holidays=custom_holidays)

    # Build features for the given datetime
    day_num = dt.weekday()
    hour = dt.hour

    # Try to estimate Prev_* values using historical medians for same day_num & hour
    grp = df_h.groupby(["Day_num", "Hour"]).agg({"PM2_5": "median", "PM10": "median", "Noise": "median"}).reset_index()
    match = grp[(grp["Day_num"] == day_num) & (grp["Hour"] == ((hour - 1) % 24))]
    if match.empty:
        # fallback to same hour
        match = grp[(grp["Day_num"] == day_num) & (grp["Hour"] == hour)]
    if match.empty:
        # last resort: global medians
        med = df_h[["PM2_5", "PM10", "Noise"]].median()
        prev_pm2_5 = med["PM2_5"]
        prev_pm10 = med["PM10"]
        prev_noise = med["Noise"]
    else:
        prev_pm2_5 = float(match["PM2_5"].values[0])
        prev_pm10 = float(match["PM10"].values[0])
        prev_noise = float(match["Noise"].values[0])

    if custom_holidays is None:
        custom_holidays = load_custom_holidays()
    is_holiday = int(dt.date() in custom_holidays)

    feat = {
        "Day_num": day_num,
        "Hour": hour,
        "Prev_PM2_5": prev_pm2_5,
        "Prev_PM10": prev_pm10,
        "Prev_Noise": prev_noise,
        "Is_Holiday": is_holiday,
    }

    FEATURES = [
        "Day_num",
        "Hour",
        "Prev_PM2_5",
        "Prev_PM10",
        "Prev_Noise",
        "Is_Holiday",
    ]

    X_pred = pd.DataFrame([feat])[FEATURES]

    # Load or use provided model
    if model is None:
        if not os.path.exists(MODEL_OUT_PATH):
            raise FileNotFoundError("Trained model not found. Train first or pass a model instance.")
        model = joblib.load(MODEL_OUT_PATH)

    # Compatibility shim: if the loaded model records feature names (sklearn's
    # `feature_names_in_`), ensure `X_pred` has those columns in the same order.
    # Fill any missing features with sensible defaults (0).
    try:
        if hasattr(model, "feature_names_in_"):
            expected = list(model.feature_names_in_)
            # Reindex X_pred to expected columns, filling missing with 0
            X_pred = X_pred.reindex(columns=expected, fill_value=0)
    except Exception:
        # If anything goes wrong, fall back to original X_pred
        pass

    pred = model.predict(X_pred)[0]
    return dict(PM2_5=float(pred[0]), PM10=float(pred[1]), Noise=float(pred[2]))


def main():
    parser = argparse.ArgumentParser(description="Train multi-output air/noise model or predict for given datetime")
    parser.add_argument("--predict", type=str, help="Datetime to predict: 'YYYY-MM-DD HH:MM' (local time)")
    parser.add_argument("--train-only", action="store_true", help="Train model and exit")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    df = load_and_merge()

    # Train model (and save predictions)
    train_and_save_model(df)

    if args.predict:
        dt = pd.to_datetime(args.predict)
        preds = predict_for_datetime(dt, df)
        print(f"\nPrediction for {dt}:")
        for k, v in preds.items():
            print(f" - {k}: {v:.2f}")


if __name__ == "__main__":
    main()

