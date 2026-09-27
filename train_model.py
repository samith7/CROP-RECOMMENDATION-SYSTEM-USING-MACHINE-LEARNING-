import os
import json
import requests
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "Crop_Recommendation.csv")
DATA_URL = "https://raw.githubusercontent.com/nileshiq/Crop-Recommendation/main/Crop_Recommendation.csv"

FEATURES = [
    "Nitrogen", "Phosphorus", "Potassium",
    "Temperature", "Humidity", "pH_Value", "Rainfall"
]

def download_dataset():
    if os.path.exists(DATA_PATH):
        return
    print("Dataset not found. Downloading...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    with open(DATA_PATH, "wb") as f:
        f.write(response.content)
    print("Dataset downloaded:", DATA_PATH)

def load_dataset():
    df = pd.read_csv(DATA_PATH)

    # Support both the downloaded mirror's column names and common dataset naming.
    rename_map = {
        "N": "Nitrogen",
        "P": "Phosphorus",
        "K": "Potassium",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "ph": "pH_Value",
        "rainfall": "Rainfall",
        "label": "Crop",
    }
    df = df.rename(columns=rename_map)

    missing = [c for c in FEATURES + ["Crop"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[FEATURES + ["Crop"]].dropna()
    return df

def main():
    download_dataset()
    df = load_dataset()

    print("\nDataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["Crop"].value_counts())

    X = df[FEATURES]
    y_text = df["Crop"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=12),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=10, probability=True))
        ])
    }

    rows = []
    fitted = {}

    print("\nTraining models...\n")
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0)
        }
        rows.append(metrics)
        fitted[name] = model

        print(
            f"{name}: "
            f"Accuracy={metrics['Accuracy']:.4f}, "
            f"Precision={metrics['Precision']:.4f}, "
            f"Recall={metrics['Recall']:.4f}, "
            f"F1={metrics['F1 Score']:.4f}"
        )

    comparison = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
    comparison.to_csv(os.path.join(RESULT_DIR, "model_comparison.csv"), index=False)

    # Use Random Forest as the final model for the application.
    final_model = fitted["Random Forest"]
    joblib.dump(final_model, os.path.join(MODEL_DIR, "crop_model.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # Detailed report and confusion matrix for Random Forest.
    rf_pred = final_model.predict(X_test)
    report = classification_report(
        y_test, rf_pred, target_names=encoder.classes_, zero_division=0
    )
    with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=encoder.classes_, yticklabels=encoder.classes_
    )
    plt.title("Random Forest - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=180)
    plt.close()

    # Feature importance.
    importance = pd.Series(
        final_model.feature_importances_, index=FEATURES
    ).sort_values(ascending=True)

    plt.figure(figsize=(9, 5))
    importance.plot(kind="barh")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "feature_importance.png"), dpi=180)
    plt.close()

    metadata = {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "number_of_crops": int(df["Crop"].nunique()),
        "features": FEATURES,
        "final_model": "Random Forest",
        "classes": list(encoder.classes_),
        "results": comparison.to_dict(orient="records")
    }
    with open(os.path.join(RESULT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nTraining complete.")
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print("\nSaved model:", os.path.join(MODEL_DIR, "crop_model.pkl"))
    print("Now run: streamlit run app.py")

if __name__ == "__main__":
    main()
