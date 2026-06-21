# Core tools for handling datasets
import pandas as pd
import os

# Modeling engines and verification metrics
import xgboost as xgb
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
import joblib

# Cloud registration tools
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

# Setup our MLflow experimentation workspace properties
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")

# Link into the Hugging Face communication engine using our stored token
hf_token = os.getenv("HF_TOKEN")
api = HfApi(token=hf_token)

Xtrain_path = "hf://datasets/mshahban/tourism-prediction-dataset/Xtrain.csv"
Xtest_path = "hf://datasets/mshahban/tourism-prediction-dataset/Xtest.csv"
ytrain_path = "hf://datasets/mshahban/tourism-prediction-dataset/ytrain.csv"
ytest_path = "hf://datasets/mshahban/tourism-prediction-dataset/ytest.csv"

# Reading directly from your local disk workspace to avoid Hugging Face URL pathing bugs
print("Loading split datasets from local runtime...")
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)
print("Local dataset assets loaded successfully.")

# Extract arrays safely down to single column series
ytrain = ytrain['ProdTaken'] if 'ProdTaken' in ytrain.columns else ytrain.squeeze()
ytest = ytest['ProdTaken'] if 'ProdTaken' in ytest.columns else ytest.squeeze()

# Compute the balance adjustment score to compensate for target imbalances
imbalance_ratio = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Instantiate our baseline XGBoost classifier
xgb_engine = xgb.XGBClassifier(scale_pos_weight=imbalance_ratio, random_state=42)

# Define our model training search matrix parameters
search_parameters = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Construct the executable scikit-learn model architecture pipeline
model_pipeline = make_pipeline(xgb_engine)

# Execute parameter search loop inside our master tracking timeline
with mlflow.start_run():
    print("Initiating hyperparameter grid tuning search...")
    # Configure cross-validation grid solver
    cv_tuner = GridSearchCV(model_pipeline, search_parameters, cv=5, n_jobs=-1)
    cv_tuner.fit(Xtrain, ytrain)

    # Stream individual parameter combinations dynamically into sub-runs
    tuning_logs = cv_tuner.cv_results_
    for idx in range(len(tuning_logs['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(tuning_logs['params'][idx])
            mlflow.log_metric("mean_test_score", tuning_logs['mean_test_score'][idx])
            mlflow.log_metric("std_test_score", tuning_logs['std_test_score'][idx])

    # Archive the optimal configuration selections to the master run
    mlflow.log_params(cv_tuner.best_params_)
    optimized_model = cv_tuner.best_estimator_

    # Apply specialized probability filtering to improve performance tracking
    probability_cap = 0.45

    train_probabilities = optimized_model.predict_proba(Xtrain)[:, 1]
    train_predictions = (train_probabilities >= probability_cap).astype(int)

    test_probabilities = optimized_model.predict_proba(Xtest)[:, 1]
    test_predictions = (test_probabilities >= probability_cap).astype(int)

    # Compile the final classification metrics dictionaries
    train_metrics = classification_report(ytrain, train_predictions, output_dict=True)
    test_metrics = classification_report(ytest, test_predictions, output_dict=True)

    # Stream final performance criteria summaries directly to the dashboard
    mlflow.log_metrics({
        "train_accuracy": train_metrics['accuracy'],
        "train_precision": train_metrics['1']['precision'],
        "train_recall": train_metrics['1']['recall'],
        "train_f1-score": train_metrics['1']['f1-score'],
        "test_accuracy": test_metrics['accuracy'],
        "test_precision": test_metrics['1']['precision'],
        "test_recall": test_metrics['1']['recall'],
        "test_f1-score": test_metrics['1']['f1-score']
    })

    # Save the optimized model artifact locally to disk
    saved_model_filename = "best_tourism_package_model_v1.joblib"
    joblib.dump(optimized_model, saved_model_filename)

    # Save artifact directly to the MLflow logging file path
    mlflow.log_artifact(saved_model_filename, artifact_path="model")
    print(f"Model successfully tracked and saved locally as: {saved_model_filename}")

    # Set repository targets to point to your Model Hub profile
    model_repo_id = "mshahban/tourism-package-model"
    model_repo_type = "model"

    # Verify repository existence before committing data
    try:
        api.repo_info(repo_id=model_repo_id, repo_type=model_repo_type)
        print(f"Model space '{model_repo_id}' already exists. Reusing target repository.")
    except RepositoryNotFoundError:
        print(f"Model space '{model_repo_id}' missing. Creating cloud asset repo...")
        api.create_repo(repo_id=model_repo_id, repo_type=model_repo_type, private=False)
        print(f"Model space '{model_repo_id}' generated successfully.")

    # Upload the trained binary up to your Hugging Face Model Space
    api.upload_file(
        path_or_fileobj=saved_model_filename,
        path_in_repo=saved_model_filename,
        repo_id=model_repo_id,
        repo_type=model_repo_type,
    )
    print("Model deployment binary synced successfully to Hugging Face Hub!")
