import os, json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATASET_PATH = os.getenv("DATASET_PATH", "dataset.csv")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "churn")
MODEL_OUTPUT = os.getenv("MODEL_OUTPUT", "model.pkl")
# test retraining github actions
df = pd.read_csv(DATASET_PATH)
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target '{TARGET_COLUMN}' not found. Columns: {list(df.columns)}")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

counts = y.value_counts()
stratify_value = y if len(counts) > 1 and counts.min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_value
)

model = RandomForestClassifier(
    n_estimators=120, max_depth=8, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
}

joblib.dump(model, MODEL_OUTPUT)
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
with open("model_metadata.json", "w") as f:
    json.dump({
        "task": "classification",
        "target": TARGET_COLUMN,
        "features": list(X.columns),
        "metrics": metrics
    }, f, indent=2)

print("Training completed successfully")
print(json.dumps(metrics, indent=2))
print("Features:", list(X.columns))
