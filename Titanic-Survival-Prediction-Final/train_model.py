"""Train and evaluate Titanic survival ML/DL models.
Run: python train_model.py
"""
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception as exc:
    XGB_AVAILABLE = False
    XGB_ERROR = str(exc)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Titanic-Dataset.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
RANDOM_STATE = 42


def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def metric_row(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}\nPut Titanic-Dataset.csv inside the data folder.")

    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + ["Survived"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Basic analysis figures.
    plt.figure(figsize=(6, 4))
    df["Survived"].value_counts().sort_index().plot(kind="bar")
    plt.title("Titanic Survival Distribution")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "survival_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    pd.crosstab(df["Sex"], df["Survived"]).plot(kind="bar")
    plt.title("Survival by Gender")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "survival_by_gender.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    pd.crosstab(df["Pclass"], df["Survived"]).plot(kind="bar")
    plt.title("Survival by Passenger Class")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "survival_by_class.png", dpi=160)
    plt.close()

    X = df[FEATURES].copy()
    y = df["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE,
            n_jobs=2, tree_method="hist"
        )

    results = []
    fitted_ml = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        results.append(metric_row(name, y_test, pred, prob))
        fitted_ml[name] = pipe

    # DL models are included when TensorFlow is installed (as required by requirements.txt).
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
        tf.random.set_seed(RANDOM_STATE)
        tf.keras.utils.set_random_seed(RANDOM_STATE)
        tf_available = True
    except Exception as exc:
        tf_available = False
        tf_error = str(exc)

    if tf_available:
        preprocessor = make_preprocessor()
        X_train_dense = preprocessor.fit_transform(X_train).astype("float32")
        X_test_dense = preprocessor.transform(X_test).astype("float32")

        def train_ann():
            model = Sequential([
                Dense(64, activation="relu", input_shape=(X_train_dense.shape[1],)),
                Dropout(0.30), Dense(32, activation="relu"), Dropout(0.20),
                Dense(16, activation="relu"), Dense(1, activation="sigmoid")
            ])
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(X_train_dense, y_train, epochs=50, batch_size=32, validation_split=0.20, verbose=0)
            return model

        ann = train_ann()
        ann_prob = ann.predict(X_test_dense, verbose=0).ravel()
        ann_pred = (ann_prob >= 0.5).astype(int)
        results.append(metric_row("ANN", y_test, ann_pred, ann_prob))
        ann.save(MODEL_DIR / "ann_model.keras")

        xtr = X_train_dense.reshape(X_train_dense.shape[0], X_train_dense.shape[1], 1)
        xte = X_test_dense.reshape(X_test_dense.shape[0], X_test_dense.shape[1], 1)
        cnn = Sequential([
            Conv1D(32, 3, activation="relu", padding="same", input_shape=(xtr.shape[1], 1)),
            MaxPooling1D(2), Conv1D(64, 3, activation="relu", padding="same"),
            Flatten(), Dense(32, activation="relu"), Dropout(0.30), Dense(1, activation="sigmoid")
        ])
        cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        cnn.fit(xtr, y_train, epochs=50, batch_size=32, validation_split=0.20, verbose=0)
        cnn_prob = cnn.predict(xte, verbose=0).ravel()
        cnn_pred = (cnn_prob >= 0.5).astype(int)
        results.append(metric_row("1D CNN", y_test, cnn_pred, cnn_prob))
        cnn.save(MODEL_DIR / "cnn_model.keras")
    else:
        print("TensorFlow is not available; ANN and 1D CNN were skipped.")
        print(f"TensorFlow error: {tf_error}")

    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    ax = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].plot(kind="bar", figsize=(13, 6))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Titanic Model Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=160)
    plt.close()

    # Deploy the best ML pipeline because it contains preprocessing + model and is easy to load in Streamlit.
    ml_results = results_df[results_df["Model"].isin(fitted_ml.keys())]
    best_ml_name = ml_results.iloc[0]["Model"]
    best_pipeline = fitted_ml[best_ml_name]
    joblib.dump(best_pipeline, MODEL_DIR / "titanic_best_model.pkl")

    best_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Not Survived", "Survived"])
    plt.yticks([0, 1], ["Not Survived", "Survived"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_ml_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    info = {
        "best_overall_model_by_f1": str(results_df.iloc[0]["Model"]),
        "deployed_model": best_ml_name,
        "deployed_model_f1": float(ml_results.iloc[0]["F1 Score"]),
        "dataset_rows": int(len(df)),
        "dataset_columns": int(df.shape[1]),
        "tensorflow_available": tf_available,
    }
    (MODEL_DIR / "best_model_info.txt").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nTraining complete.")
    print(results_df.to_string(index=False))
    print(f"\nSaved deployable model: {MODEL_DIR / 'titanic_best_model.pkl'}")

if __name__ == "__main__":
    main()
