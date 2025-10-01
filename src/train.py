import os, json, argparse
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def build_pipeline(numeric_cols, categorical_cols, n_estimators=300, max_depth=8, random_state=42):
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols)
        ],
        remainder="drop"
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced"
    )
    return Pipeline(steps=[("pre", pre), ("clf", model)])

def main(data_path, out_dir, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    assert "Churn" in df.columns, "No se encontró la columna objetivo 'Churn'. Ejecuta el preprocess primero."

    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])

    # detecta tipos
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # MLflow local (sqlite + carpeta ./mlruns)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("telco-churn")

    with mlflow.start_run():
        pipe = build_pipeline(numeric_cols, categorical_cols)
        pipe.fit(Xtr, ytr)

        # predicciones y métricas
        yhat = pipe.predict(Xte)
        acc = accuracy_score(yte, yhat)
        prec = precision_score(yte, yhat, zero_division=0)
        rec = recall_score(yte, yhat, zero_division=0)
        f1 = f1_score(yte, yhat, zero_division=0)

        # AUC (si el modelo soporta predict_proba)
        try:
            proba = pipe.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, proba)
        except Exception:
            auc = None

        # log params + metrics a MLflow
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        if auc is not None:
            mlflow.log_metric("roc_auc", auc)

        # registra el modelo en el registry
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name="telco-churn"
        )

        # guarda artefactos locales para DVC
        os.makedirs(out_dir, exist_ok=True)
        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        if auc is not None:
            metrics["roc_auc"] = auc
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        joblib.dump(pipe, os.path.join(out_dir, "model.pkl"))

        print("Métricas:", metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/telco_clean.csv")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    print("Un cambio en train.py")
    main(args.data, args.out)
