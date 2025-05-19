import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, matthews_corrcoef, balanced_accuracy_score
)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "croprecommendation_preprocessing.csv")
    data = pd.read_csv(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("label", axis=1),
        data["label"],
        random_state=42,
        test_size=0.2
    )
    input_example = X_train[0:5]
    dataset = mlflow.data.from_pandas(data)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 257
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 13

    with mlflow.start_run():

        mlflow.log_input(dataset, context="dataset")

        rf_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        }

        mlflow.log_params(rf_params)

        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
        log_loss_score = log_loss(y_test, y_pred_proba)

        mattthews_corrcoef = matthews_corrcoef(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)


        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "log_loss": log_loss_score,
            "mattthews_corrcoef": mattthews_corrcoef,
            "balanced_accuracy": balanced_accuracy
        })

        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_score", train_score)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
            )