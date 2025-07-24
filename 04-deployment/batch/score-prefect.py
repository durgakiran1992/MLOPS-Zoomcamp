#!/usr/bin/env python
# coding: utf-8

import os
import sys
import uuid
from datetime import datetime
import pandas as pd
import mlflow

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context
from dateutil.relativedelta import relativedelta


@task
def generate_uuids(n):
    return [str(uuid.uuid4()) for _ in range(n)]


@task
def read_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df['ride_id'] = generate_uuids.fn(len(df))  # Call underlying function inside task
    return df


@task
def prepare_dictionaries(df: pd.DataFrame):
    df = df.copy()
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df[['PU_DO', 'trip_distance']].to_dict(orient='records')


@task
def load_model(run_id: str):
    model_uri = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(model_uri)
    return model


@task
def predict(model, dicts):
    return model.predict(dicts)


@task
def save_results(df, y_pred, run_id, output_file):
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'lpep_pickup_datetime': df['lpep_pickup_datetime'],
        'PULocationID': df['PULocationID'],
        'DOLocationID': df['DOLocationID'],
        'actual_duration': df['duration'],
        'predicted_duration': y_pred,
        'diff': df['duration'] - y_pred,
        'model_version': run_id
    })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_result.to_parquet(output_file, index=False)
    print(f"✅ Predictions saved to: {output_file}")


def get_paths(run_date, taxi_type, run_id):
    prev_month = run_date - relativedelta(months=1)
    year, month = prev_month.year, prev_month.month

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output1/{taxi_type}/{year:04d}-{month:02d}.parquet'

    return input_file, output_file


@flow(name="ride-duration-prediction")
def ride_duration_prediction(taxi_type: str, run_id: str, run_date: datetime = None):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger = get_run_logger()

    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time

    input_file, output_file = get_paths(run_date, taxi_type, run_id)

    logger.info(f"🚕 Processing {input_file}")

    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)
    model = load_model(run_id)
    y_pred = predict(model, dicts)
    save_results(df, y_pred, run_id, output_file)


def run():
    taxi_type = sys.argv[1]           # e.g. 'green'
    year = int(sys.argv[2])           # e.g. 2021
    month = int(sys.argv[3])          # e.g. 3
    run_id = sys.argv[4]              # MLflow run_id

    ride_duration_prediction(
        taxi_type=taxi_type,
        run_id=run_id,
        run_date=datetime(year=year, month=month, day=1)
    )


if __name__ == '__main__':
    run()
