import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
import tensorflow as tf

# Always work from the folder where app.py is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory:", os.getcwd())

app = Flask(__name__)
keras = tf.keras

# Lazy loading for Render stability
model = None
pipeline = None


def load_assets():
    global model, pipeline

    if pipeline is None:
        print("Loading preprocess pipeline...")
        pipeline = joblib.load("preprocess_pipeline.pkl")

    if model is None:
        print("Loading Keras model...")
        model = keras.models.load_model("fires_model.keras", compile=False)

    return model, pipeline


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return "OK", 200


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        try:
            model, pipeline = load_assets()

            data = {
                "longitude": [float(request.form["longitude"])],
                "latitude": [float(request.form["latitude"])],
                "month": [request.form["month"].strip()],
                "day": [request.form["day"].strip()],
                "avg_temp": [float(request.form["avg_temp"])],
                "max_temp": [float(request.form["max_temp"])],
                "max_wind_speed": [float(request.form["max_wind_speed"])],
                "avg_wind": [float(request.form["avg_wind"])]
            }

            print("Received form data:", data)

            input_df = pd.DataFrame(data)
            print("Input DataFrame:")
            print(input_df)

            input_prepared = pipeline.transform(input_df)

            if hasattr(input_prepared, "toarray"):
                input_prepared = input_prepared.toarray()

            print("Transformed input shape:", input_prepared.shape)

            pred_log = model.predict(input_prepared, verbose=0)[0][0]
            pred_real = float(np.exp(pred_log) - 1)

            print("Prediction (log scale):", pred_log)
            print("Prediction (real scale):", pred_real)

            return render_template("result.html", prediction=round(pred_real, 2))

        except Exception as e:
            print("Prediction error:", repr(e))
            return f"Model failed: {e}", 500

    return render_template("prediction.html")


if __name__ == "__main__":
    app.run(debug=True)