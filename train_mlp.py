import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import tensorflow as tf
from tensorflow import keras


# 1. Load data
fires = pd.read_csv("C:\\Users\\dkqle\\sanbul-project\\sanbul2district-divby100.csv", sep=",")

# 2. Log transform target
fires["burned_area"] = np.log(fires["burned_area"] + 1)

# 3. Basic outputs
print("===== HEAD =====")
print(fires.head())

print("\n===== INFO =====")
print(fires.info())

print("\n===== DESCRIBE =====")
print(fires.describe())

print("\n===== MONTH COUNTS =====")
print(fires["month"].value_counts())

print("\n===== DAY COUNTS =====")
print(fires["day"].value_counts())

# 4. Histogram
fires.hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.savefig("histograms.png")
plt.close()

# 5. Stratified split by month
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_idx]
    strat_test_set = fires.loc[test_idx]

print("\n===== TEST MONTH RATIO =====")
print(strat_test_set["month"].value_counts() / len(strat_test_set))

print("\n===== OVERALL MONTH RATIO =====")
print(fires["month"].value_counts() / len(fires))

# 6. Separate features and labels
train_features = strat_train_set.drop("burned_area", axis=1)
train_labels = strat_train_set["burned_area"].copy()

test_features = strat_test_set.drop("burned_area", axis=1)
test_labels = strat_test_set["burned_area"].copy()

# 7. Preprocessing pipeline
num_attribs = ["longitude", "latitude", "avg_temp", "max_temp", "max_wind_speed", "avg_wind"]
cat_attribs = ["month", "day"]

num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs)
])

train_prepared = full_pipeline.fit_transform(train_features)
test_prepared = full_pipeline.transform(test_features)

# Convert sparse matrix to dense if needed
train_prepared = train_prepared.toarray() if hasattr(train_prepared, "toarray") else train_prepared
test_prepared = test_prepared.toarray() if hasattr(test_prepared, "toarray") else test_prepared

# 8. Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    train_prepared, train_labels.values, test_size=0.2, random_state=42
)

# 9. Build MLP model
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Input(shape=[X_train.shape[1]]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])

model.summary()

# 10. Compile and train
model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.SGD(learning_rate=1e-3)
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_data=(X_valid, y_valid),
    verbose=1
)

print("Current working directory:", os.getcwd())

print("Saving training history plot...")
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.savefig("training_history.png")
plt.close()

print("Saving model...")
model.save("fires_model.keras")

print("Saving pipeline...")
joblib.dump(full_pipeline, "preprocess_pipeline.pkl")

print("Saving histograms...")
fires.hist(bins=20, figsize=(12, 8))
plt.tight_layout()
plt.savefig("histograms.png")
plt.close()

print("Files now in folder:")
print(os.listdir())

print("Done. Model and pipeline saved.")