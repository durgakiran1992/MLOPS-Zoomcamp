import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
REGISTERED_MODEL_NAME = "random-forest-best"  # you can change this name as needed
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run() as run:
        new_params = {param: int(params[param]) for param in RF_PARAMS}

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))

        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        # Save model to artifact location for registration
        mlflow.sklearn.log_model(rf, artifact_path="model")

        return run.info.run_id, test_rmse


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    # Step 1: Get best runs from hyperopt experiment
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    top_runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    # Step 2: Train & evaluate top models on test set
    run_scores = []
    for run in top_runs:
        run_id, test_rmse = train_and_log_model(data_path=data_path, params=run.data.params)
        run_scores.append((run_id, test_rmse))

    # Step 3: Select best model (lowest test_rmse)
    best_run_id, best_rmse = sorted(run_scores, key=lambda x: x[1])[0]
    print(f"✅ Best run: {best_run_id}, test RMSE: {best_rmse:.4f}")

    # Step 4: Register model
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)


if __name__ == '__main__':
    run_register_model()
