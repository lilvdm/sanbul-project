import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())

keras = tf.keras
app = Flask(__name__)

model = keras.models.load_model("fires_model.keras")
pipeline = joblib.load("preprocess_pipeline.pkl")


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        data = {
            "longitude": [float(request.form["longitude"])],
            "latitude": [float(request.form["latitude"])],
            "month": [request.form["month"]],
            "day": [request.form["day"]],
            "avg_temp": [float(request.form["avg_temp"])],
            "max_temp": [float(request.form["max_temp"])],
            "max_wind_speed": [float(request.form["max_wind_speed"])],
            "avg_wind": [float(request.form["avg_wind"])]
        }

        input_df = pd.DataFrame(data)
        input_prepared = pipeline.transform(input_df)
        input_prepared = input_prepared.toarray() if hasattr(input_prepared, "toarray") else input_prepared

        pred_log = model.predict(input_prepared)[0][0]
        pred_real = np.exp(pred_log) - 1

        return render_template("result.html", prediction=round(pred_real, 2))

    return render_template("prediction.html")


if __name__ == "__main__":
    app.run(debug=True)