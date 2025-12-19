import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from mlflow.models.signature import infer_signature


# Fungsi untuk load data
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


# Fungsi untuk melatih model
def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(
        base_dir, "water_potability_preprocessing", "train_potability.csv"
    )
    test_path = os.path.join(
        base_dir, "water_potability_preprocessing", "test_potability.csv"
    )

    print("[INFO] Loading data...")
    train_df = load_data(train_path)
    test_df = load_data(test_path)

    X_train = train_df.drop("Potability", axis=1)
    y_train = train_df["Potability"]
    X_test = test_df.drop("Potability", axis=1)
    y_test = test_df["Potability"]

    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        print("[INFO] Training model...")

        n_estimators = 200
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)

        print("[INFO] All process executed successfully.")


if __name__ == "__main__":
    train_model()
