import argparse
import os
import pickle
import pandas as pd

# Categorical features used in training
categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def prepare_features(df, dv):
    dicts = df[categorical].to_dict(orient='records')
    return dv.transform(dicts)


def run(year, month):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    os.makedirs('output', exist_ok=True)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(input_file)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    X_val = prepare_features(df, dv)
    y_pred = lr.predict(X_val)

    print(f"Mean predicted duration: {y_pred.mean():.2f} minutes")

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        index=False
    )

    print(f"✅ Saved predictions to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2023)')
    parser.add_argument('--month', type=int, required=True, help='Month (e.g., 3 for March)')
    args = parser.parse_args()

    run(args.year, args.month)
