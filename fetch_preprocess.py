"""Fetch and preprocess ThingSpeak data for AirNoise project.

This script fetches the channel feeds from ThingSpeak (JSON), maps the
first three fields to `PM2_5`, `PM10`, `Noise` (attempts to be robust),
adds time-based features, a holiday flag, and a simple location
flag for "hostel near main road". It saves cleaned data to
`dataset/cleaned_ts_data.csv` and can optionally merge with
`dataset/synthetic_data.csv` and save `dataset/merged_train_data.csv`.

Holidays are loaded from dataset/custom_holidays.csv (date,name format).

Usage (simple):
    python fetch_preprocess.py

The channel id and read API key are embedded as requested.
"""

import os
import requests
import pandas as pd
from datetime import datetime

# ================= CONFIG =================
THINGSPEAK_CHANNEL_ID = 3130355
THINGSPEAK_READ_API_KEY = "1QXEREJK6PQLCLLI"
OUT_PATH = os.path.join("dataset", "cleaned_ts_data.csv")
MERGED_OUT_PATH = os.path.join("dataset", "merged_train_data.csv")
SYNTHETIC_PATH = os.path.join("dataset", "synthetic_data.csv")
CUSTOM_HOLIDAYS_PATH = os.path.join("dataset", "custom_holidays.csv")


def load_custom_holidays():
    """Load custom holidays from CSV into a set of dates."""
    holiday_set = set()
    if os.path.exists(CUSTOM_HOLIDAYS_PATH):
        try:
            custom_df = pd.read_csv(CUSTOM_HOLIDAYS_PATH, parse_dates=["date"])
            custom_set = set(custom_df["date"].dt.date)
            holiday_set.update(custom_set)
        except Exception as e:
            print(f"Warning: Could not load custom holidays: {e}")
    return holiday_set

def fetch_thingspeak(channel_id: int, read_api_key: str, results: int = 8000):
	url = (
		f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
		f"?api_key={read_api_key}&results={results}"
	)
	resp = requests.get(url, timeout=20)
	resp.raise_for_status()
	return resp.json()


def parse_thingspeak_json(js: dict) -> pd.DataFrame:
	feeds = js.get("feeds", [])
	if not feeds:
		return pd.DataFrame()

	# Convert list of feeds to DataFrame
	df = pd.DataFrame(feeds)

	# Standardize timestamp field
	if "created_at" in df.columns:
		df = df.rename(columns={"created_at": "Timestamp"})

	# Map the first three fieldN columns to PM2_5, PM10, Noise if present
	field_cols = [c for c in df.columns if c.startswith("field")]
	mapping = {}
	if len(field_cols) >= 1:
		mapping[field_cols[0]] = "PM2_5"
	if len(field_cols) >= 2:
		mapping[field_cols[1]] = "PM10"
	if len(field_cols) >= 3:
		mapping[field_cols[2]] = "Noise"

	if mapping:
		df = df.rename(columns=mapping)

	# Keep only useful columns if they exist
	cols_keep = [c for c in ["Timestamp", "PM2_5", "PM10", "Noise"] if c in df.columns]
	df = df[cols_keep].copy()

	# Parse timestamps
	if "Timestamp" in df.columns:
		df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
	df = df.dropna(subset=[c for c in ["Timestamp"] if c in df.columns])

	# Convert sensor columns to numeric
	for c in ["PM2_5", "PM10", "Noise"]:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors="coerce")

	df = df.dropna().reset_index(drop=True)
	return df


def add_time_and_context_features(df: pd.DataFrame, custom_holidays=None) -> pd.DataFrame:
	df = df.copy()
	df["Day"] = df["Timestamp"].dt.day_name()
	df["Hour"] = df["Timestamp"].dt.hour
	df["Day_num"] = df["Timestamp"].dt.weekday  # Monday=0

	# Holiday detection: custom CSV
	if custom_holidays is None:
		custom_holidays = load_custom_holidays()
	df["Is_Holiday"] = df["Timestamp"].dt.date.apply(lambda d: d in custom_holidays)

	return df


def save_cleaned(df: pd.DataFrame, out_path: str = OUT_PATH):
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	df.to_csv(out_path, index=False)
	print(f"Saved cleaned ThingSpeak data to {out_path}")


def merge_with_synthetic(cleaned_path: str = OUT_PATH, synthetic_path: str = SYNTHETIC_PATH, out_path: str = MERGED_OUT_PATH):
	if not os.path.exists(synthetic_path):
		print("Synthetic data not found, skipping merge.")
		return
	df_clean = pd.read_csv(cleaned_path, parse_dates=["Timestamp"]) if os.path.exists(cleaned_path) else pd.DataFrame()
	df_synth = pd.read_csv(synthetic_path, parse_dates=["Timestamp"]) if os.path.exists(synthetic_path) else pd.DataFrame()

	# Ensure synthetic has the same context features
	custom_holidays = load_custom_holidays()
	if not df_synth.empty:
		if "Day_num" not in df_synth.columns:
			df_synth["Day_num"] = df_synth["Timestamp"].dt.weekday
		if "Is_Holiday" not in df_synth.columns:
			df_synth["Is_Holiday"] = df_synth["Timestamp"].dt.date.apply(lambda d: d in custom_holidays)

	if df_clean.empty and df_synth.empty:
		print("No data to merge.")
		return

	if df_clean.empty:
		merged = df_synth
	elif df_synth.empty:
		merged = df_clean
	else:
		merged = pd.concat([df_synth, df_clean], ignore_index=True).sort_values("Timestamp").reset_index(drop=True)

	merged.to_csv(out_path, index=False)
	print(f"Saved merged training data to {out_path}")


def main():
	print("Fetching ThingSpeak channel... (this may take a few seconds)")
	try:
		js = fetch_thingspeak(THINGSPEAK_CHANNEL_ID, THINGSPEAK_READ_API_KEY)
		df = parse_thingspeak_json(js)
		if df.empty:
			print("No feeds returned from ThingSpeak or fields not present.")
			return
		# add_time_and_context_features no longer accepts a hostel/location flag;
		# we only add time and holiday context here.
		df = add_time_and_context_features(df)
		save_cleaned(df, OUT_PATH)
		merge_with_synthetic(OUT_PATH, SYNTHETIC_PATH, MERGED_OUT_PATH)
	except Exception as e:
		print("Error fetching or processing ThingSpeak data:", e)


if __name__ == "__main__":
	main()

