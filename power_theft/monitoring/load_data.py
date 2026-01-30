import pandas as pd
import numpy as np
import os
from .utils import get_data_path

def get_data_file(filename):
    """Get full path to data file"""
    return os.path.join(get_data_path(), filename)

def prepare_training_data():
    meters, meter_readings, transformer = load_data()
    
    # Merge meter readings with transformer readings
    # Aggregate meter readings by transformer and timestamp
    meter_agg = meter_readings.groupby(['transformer_id', 'timestamp']).agg({
        'energy_consumed_kwh': 'sum',
        'energy_purchased_kwh': 'sum'
    }).reset_index()
    
    # Merge with transformer data
    df = pd.merge(transformer, meter_agg, on=['transformer_id', 'timestamp'], how='inner')
    
    # Create theft flag based on loss ratio threshold
    df['theft_flag'] = (df['loss_ratio'] > 0.15).astype(int)
    
    return df

def load_data():
    try:
        meters = pd.read_csv(get_data_file('datalarge_meters.csv'))
        meter_readings = pd.read_csv(
            get_data_file('datalarge_meter_readings.csv'),
            parse_dates=["timestamp"]
        )
        transformer = pd.read_csv(
            get_data_file('datalarge_transformer_readings.csv'),
            parse_dates=["timestamp"]
        )
        return meters, meter_readings, transformer
    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Please ensure data files are uploaded to the server")
        # Return empty DataFrames for now
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def create_features(transformer_df):
    df = transformer_df.copy()

    # ---- Time features ----
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_peak_hour"] = df["hour"].between(18, 22).astype(int)

    # ---- Rolling statistics (key for theft detection) ----
    df = df.sort_values(["transformer_id", "timestamp"])

    df["rolling_loss_24h"] = (
        df.groupby("transformer_id")["loss_ratio"]
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_loss_7d"] = (
        df.groupby("transformer_id")["loss_ratio"]
        .rolling(168)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_supply_24h"] = (
        df.groupby("transformer_id")["energy_supplied_kwh"]
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df = df.dropna()

    return df
