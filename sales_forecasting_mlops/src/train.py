
# Data manipulation
import pandas as pd
import numpy as np
from pathlib import Path

# Preprocessing and modeling
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model serialization and experiment tracking
import joblib
import mlflow
import mlflow.sklearn

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SuperKart-Sales-Forecasting-Experiment")

# Load train/test data from the previous workflow job's artifact
ARTIFACT_DIR = Path("sales_forecasting_mlops/artifacts")

Xtrain = pd.read_csv(ARTIFACT_DIR / "Xtrain.csv")
Xtest = pd.read_csv(ARTIFACT_DIR / "Xtest.csv")
ytrain = pd.read_csv(ARTIFACT_DIR / "ytrain.csv").squeeze("columns")
ytest = pd.read_csv(ARTIFACT_DIR / "ytest.csv").squeeze("columns")

print("Training and testing artifacts loaded.")
print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)

# Identify feature types
numeric_features = Xtrain.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = Xtrain.select_dtypes(
    exclude=["number"]
).columns.tolist()

# Define preprocessing
preprocessor = make_column_transformer(
    ("passthrough", numeric_features),
    (
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    )
)

# Define Random Forest regression model
rf_model = RandomForestRegressor(
    random_state=42,
    n_jobs=1
)

# Build pipeline
model_pipeline = make_pipeline(preprocessor, rf_model)

# Define hyperparameter grid
param_grid = {
    "randomforestregressor__n_estimators": [100, 200],
    "randomforestregressor__max_depth": [10, 20, None],
    "randomforestregressor__min_samples_split": [2, 5]
}

# Create output directory
MODEL_DIR = Path("sales_forecasting_mlops/artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Start MLflow run
with mlflow.start_run(run_name="SuperKart_RandomForest_Tuning"):

    # Hyperparameter tuning
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination as a nested run
    results = grid_search.cv_results_

    for i, params in enumerate(results["params"]):
        with mlflow.start_run(
            nested=True,
            run_name=f"Trial_{i + 1}"
        ):
            mlflow.log_params(params)
            mlflow.log_metric(
                "cv_MAE",
                float(-results["mean_test_score"][i])
            )
            mlflow.log_metric(
                "cv_MAE_std",
                float(results["std_test_score"][i])
            )

    # Retrieve best model and log best parameters
    best_model = grid_search.best_estimator_

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric(
        "best_cv_MAE",
        float(-grid_search.best_score_)
    )

    # Generate predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Evaluate regression performance
    metrics = {
        "train_MAE": mean_absolute_error(ytrain, y_pred_train),
        "test_MAE": mean_absolute_error(ytest, y_pred_test),
        "train_RMSE": np.sqrt(
            mean_squared_error(ytrain, y_pred_train)
        ),
        "test_RMSE": np.sqrt(
            mean_squared_error(ytest, y_pred_test)
        ),
        "train_R2": r2_score(ytrain, y_pred_train),
        "test_R2": r2_score(ytest, y_pred_test)
    }

    # Log metrics to MLflow
    mlflow.log_metrics({
        name: float(value)
        for name, value in metrics.items()
    })

    # Save best model for deployment
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    # Log saved model as an MLflow artifact
    mlflow.log_artifact(
        str(model_path),
        artifact_path="model"
    )

    # Log model with MLflow
    mlflow.sklearn.log_model(
        best_model,
        name="superkart_sales_model"
    )

    # Print results
    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"\nBest model saved to: {model_path}")
