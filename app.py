"""
Flask web UI for AirNoise prediction and actual data lookup.

Users input a date and time. The app checks if that datetime exists in the dataset
and returns actual values; otherwise, it predicts using the trained model.
"""

import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
from train_model import load_custom_holidays, load_and_merge, predict_for_datetime

app = Flask(__name__)

# Load dataset once at startup
try:
    df_history = load_and_merge()
    print(f"Loaded {len(df_history)} rows of training data")
except Exception as e:
    print(f"Warning: Could not load dataset: {e}")
    df_history = pd.DataFrame()

# Load model once at startup
MODEL_PATH = os.path.join("models", "multi_model.joblib")
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query_data():
    """
    Query endpoint: user sends a datetime, returns actual or predicted values.
    Expected JSON: {"datetime": "YYYY-MM-DD HH:MM"}
    """
    try:
        data = request.json
        datetime_str = data.get("datetime", "").strip()
        
        if not datetime_str:
            return jsonify({"error": "Missing datetime"}), 400
        
        # Parse datetime
        dt = pd.to_datetime(datetime_str)
        
        # Check if exact datetime exists in dataset
        if not df_history.empty:
            # Round to nearest hour for matching
            df_history["Timestamp_rounded"] = df_history["Timestamp"].dt.floor("H")
            dt_rounded = dt.floor("H")
            
            match = df_history[df_history["Timestamp_rounded"] == dt_rounded]
            
            if not match.empty:
                # Found actual data
                row = match.iloc[0]
                return jsonify({
                    "source": "actual",
                    "datetime": str(dt),
                    "PM2_5": round(float(row["PM2_5"]), 2),
                    "PM10": round(float(row["PM10"]), 2),
                    "Noise": round(float(row["Noise"]), 2),
                })
        
        # If not in dataset, predict
        if model is None:
            return jsonify({"error": "Model not loaded. Train the model first."}), 500
        
        preds = predict_for_datetime(dt, df_history, model=model)
        return jsonify({
            "source": "predicted",
            "datetime": str(dt),
            "PM2_5": round(preds["PM2_5"], 2),
            "PM10": round(preds["PM10"], 2),
            "Noise": round(preds["Noise"], 2),
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/info", methods=["GET"])
def info():
    """Return info about the loaded dataset and model."""
    return jsonify({
        "dataset_rows": len(df_history),
        "model_loaded": model is not None,
        "date_range": f"{df_history['Timestamp'].min()} to {df_history['Timestamp'].max()}" if not df_history.empty else "N/A"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
