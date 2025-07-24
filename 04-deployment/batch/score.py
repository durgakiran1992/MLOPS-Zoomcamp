#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import sys

import uuid
import pickle

from datetime import datetime

import pandas as pd

import mlflow

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

from dateutil.relativedelta import relativedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


# In[2]:


# year = 2021
# month = 3
# taxi_type = 'green'

# input_file = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2021-02.parquet'

# output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# RUN_ID = os.getenv('RUN_ID', '5e1a8d1da960432b8a921ffdec3965f7')


# In[3]:


def generate_uuids(n):
    return [str(uuid.uuid4()) for _ in range(n)]


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df['ride_id'] = generate_uuids(len(df))
    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    dicts = df[['PU_DO', 'trip_distance']].to_dict(orient='records')
    return dicts


# In[4]:


def load_model(run_id):
    model_uri = f'runs:/{run_id}/model'
    model = mlflow.pyfunc.load_model(model_uri)
    return model

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

@task
def apply_model(input_file, run_id, output_file):
    logger = get_run_logger()

    logger.info(f'reading the data from {input_file}...')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    logger.info(f'loading the model with RUN_ID={run_id}...')
    model = load_model(run_id)

    logger.info(f'applying the model...')
    y_pred = model.predict(dicts)

    logger.info(f'saving the result to {output_file}...')

    save_results(df, y_pred, run_id, output_file)
    return output_file

def get_paths(run_date, taxi_type, run_id):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month 

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'


    output_file = f'output1/{taxi_type}/{year:04d}-{month:02d}.parquet'

    return input_file, output_file

@flow
def ride_duration_prediction(
        taxi_type: str,
        run_id: str,
        run_date: datetime = None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time
    
    input_file, output_file = get_paths(run_date, taxi_type, run_id)

    apply_model(
        input_file=input_file,
        run_id=run_id,
        output_file=output_file
    )


def run():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # or your actual URI

    taxi_type = sys.argv[1] # 'green'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3

    run_id = sys.argv[4] # '5e1a8d1da960432b8a921ffdec3965f7'

    ride_duration_prediction(
        taxi_type=taxi_type,
        run_id=run_id,
        run_date=datetime(year=year, month=month, day=1)
    )


if __name__ == '__main__':
    run()


# In[ ]:


# get_ipython().system('ls output/green/')


# In[ ]:




