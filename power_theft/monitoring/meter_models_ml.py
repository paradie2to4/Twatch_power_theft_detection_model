from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def meter_random_forest_model():
    """Random Forest model for meter-level theft detection"""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )

def meter_isolation_forest():
    """Isolation Forest for unsupervised anomaly detection"""
    return IsolationForest(
        contamination=0.1,  # Expected proportion of anomalies
        random_state=42,
        n_estimators=100
    )

def meter_logistic_regression():
    """Logistic Regression for interpretable theft detection"""
    return LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )

# Feature sets for different models
METER_FEATURES_BASIC = [
    "energy_consumed_kwh",
    "hour",
    "weekday", 
    "is_peak_hour",
    "is_weekend",
    "consumption_24h_avg",
    "consumption_7d_avg",
    "consumption_change_1h",
    "consumption_change_24h"
]

METER_FEATURES_ADVANCED = [
    "energy_consumed_kwh",
    "hour",
    "weekday",
    "is_peak_hour", 
    "is_weekend",
    "consumption_24h_avg",
    "consumption_7d_avg",
    "consumption_change_1h",
    "consumption_change_24h",
    "consumption_stability_24h",
    "unusual_hour_consumption",
    "weekend_weekday_ratio"
]

METER_FEATURES_ELECTRICAL = [
    "energy_consumed_kwh",
    "voltage_v",
    "current_a", 
    "power_factor",
    "apparent_power_va",
    "voltage_anomaly",
    "current_anomaly",
    "pf_anomaly"
]

METER_FEATURES_FULL = METER_FEATURES_ADVANCED + [
    "consumption_deviation_pct",
    "deviation_24h_avg",
    "has_voltage_data",
    "has_current_data",
    "has_pf_data"
] + METER_FEATURES_ELECTRICAL
