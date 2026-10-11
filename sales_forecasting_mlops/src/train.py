from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ARTIFACT_DIR = Path("sales_forecasting_mlops/artifacts")
MLFLOW_DIR = Path("sales_forecasting_mlops/mlruns")
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"

def main():
    for name in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
        if not (ARTIFACT_DIR / name).exists():
            raise FileNotFoundError(f"Missing training artifact: {name}")

    Xtrain = pd.read_csv(ARTIFACT_DIR / "Xtrain.csv")
    Xtest = pd.read_csv(ARTIFACT_DIR / "Xtest.csv")
    ytrain = pd.read_csv(ARTIFACT_DIR / "ytrain.csv").squeeze("columns")
    ytest = pd.read_csv(ARTIFACT_DIR / "ytest.csv").squeeze("columns")

    numeric_features = Xtrain.select_dtypes(
        include=["number"]
    ).columns.tolist()
    categorical_features = Xtrain.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    preprocessor = make_column_transformer(
        ("passthrough", numeric_features),
        (
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    )

    pipeline = make_pipeline(
        preprocessor,
        RandomForestRegressor(random_state=42, n_jobs=1)
    )

    param_grid = {
        "randomforestregressor__n_estimators": [100, 200],
        "randomforestregressor__max_depth": [10, 20, None],
        "randomforestregressor__min_samples_split": [2, 5],
    }

    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())
    mlflow.set_experiment("SuperKart-Sales-Forecasting-Experiment")

    with mlflow.start_run(run_name="SuperKart_RandomForest_Tuning"):
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1
        )
        search.fit(Xtrain, ytrain)

        best_model = search.best_estimator_
        train_pred = best_model.predict(Xtrain)
        test_pred = best_model.predict(Xtest)

        metrics = {
            "train_MAE": mean_absolute_error(ytrain, train_pred),
            "test_MAE": mean_absolute_error(ytest, test_pred),
            "train_RMSE": np.sqrt(mean_squared_error(ytrain, train_pred)),
            "test_RMSE": np.sqrt(mean_squared_error(ytest, test_pred)),
            "train_R2": r2_score(ytrain, train_pred),
            "test_R2": r2_score(ytest, test_pred),
            "best_cv_MAE": -search.best_score_,
        }

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({
            key: float(value) for key, value in metrics.items()
        })

        for i, params in enumerate(search.cv_results_["params"]):
            with mlflow.start_run(
                nested=True, run_name=f"Trial_{i + 1}"
            ):
                mlflow.log_params(params)
                mlflow.log_metric(
                    "cv_MAE",
                    float(-search.cv_results_["mean_test_score"][i])
                )

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.sklearn.log_model(
            best_model,
            name="superkart_sales_model"
        )

        print("\nBest parameters:")
        print(search.best_params_)
        print("\nEvaluation metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print("\nSaved model:", MODEL_PATH)

if __name__ == "__main__":
    main()
