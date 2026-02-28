import pandas as pd
import numpy as np

def create_meter_features(df):
    """
    Create features for individual power meter theft detection
    """
    df = df.sort_values(["meter_id", "timestamp"])
    
    # ---- Time features ----
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_peak_hour"] = df["hour"].between(18, 22).astype(int)
    df["is_off_peak"] = df["hour"].between(0, 6).astype(int)
    
    # ---- Consumption patterns ----
    # Zero consumption detection
    df["is_zero_consumption"] = (df["energy_consumed_kwh"] <= 0.01).astype(int)
    
    # Rolling averages for consumption patterns
    df["consumption_24h_avg"] = (
        df.groupby("meter_id")["energy_consumed_kwh"]
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    df["consumption_7d_avg"] = (
        df.groupby("meter_id")["energy_consumed_kwh"]
        .rolling(168)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # Consumption deviation from expected
    if "expected_consumption_kwh" in df.columns:
        df["consumption_deviation_abs"] = (
            df["expected_consumption_kwh"] - df["energy_consumed_kwh"]
        ).abs()
        
        df["consumption_deviation_pct"] = (
            df["consumption_deviation_abs"] / df["expected_consumption_kwh"].replace(0, np.nan)
        ).fillna(0) * 100
        
        # Rolling deviation patterns
        df["deviation_24h_avg"] = (
            df.groupby("meter_id")["consumption_deviation_pct"]
            .rolling(24)
            .mean()
            .reset_index(level=0, drop=True)
        )
    
    # ---- Electrical parameter features ----
    if "voltage_v" in df.columns and "current_a" in df.columns:
        # Calculate apparent power if not provided
        df["apparent_power_va"] = df["voltage_v"] * df["current_a"]
        
        # Voltage anomalies
        df["voltage_anomaly"] = (
            (df["voltage_v"] < 200) | (df["voltage_v"] > 250)
        ).astype(int)
        
        # Current anomalies (unusually high or low)
        df["current_anomaly"] = (
            (df["current_a"] < 0.1) | (df["current_a"] > 100)
        ).astype(int)
        
        # Power factor anomalies
        if "power_factor" in df.columns:
            df["pf_anomaly"] = (
                (df["power_factor"] < 0.7) | (df["power_factor"] > 1.0)
            ).astype(int)
    
    # ---- Pattern detection features ----
    # Consistent low consumption pattern
    df["consumption_stability_24h"] = (
        df.groupby("meter_id")["energy_consumed_kwh"]
        .rolling(24)
        .std()
        .reset_index(level=0, drop=True)
    )
    
    # Sudden consumption drops
    df["consumption_change_1h"] = (
        df.groupby("meter_id")["energy_consumed_kwh"]
        .pct_change(1)
        .fillna(0)
    )
    
    df["consumption_change_24h"] = (
        df.groupby("meter_id")["energy_consumed_kwh"]
        .pct_change(24)
        .fillna(0)
    )
    
    # ---- Time-based anomaly features ----
    # Consumption during unusual hours
    df["unusual_hour_consumption"] = (
        (df["energy_consumed_kwh"] > 0) & 
        df["is_off_peak"]
    ).astype(int)
    
    # Weekend vs weekday consumption ratio
    weekend_avg = df[df["is_weekend"] == 1].groupby("meter_id")["energy_consumed_kwh"].transform("mean")
    weekday_avg = df[df["is_weekend"] == 0].groupby("meter_id")["energy_consumed_kwh"].transform("mean")
    
    df["weekend_weekday_ratio"] = weekend_avg / weekday_avg.replace(0, np.nan)
    df["weekend_weekday_ratio"] = df["weekend_weekday_ratio"].fillna(1.0)
    
    # ---- Data quality features ----
    df["has_voltage_data"] = df["voltage_v"].notna().astype(int)
    df["has_current_data"] = df["current_a"].notna().astype(int)
    df["has_pf_data"] = df["power_factor"].notna().astype(int)
    
    # Remove rows with insufficient data for feature calculation
    df = df.dropna(subset=["consumption_24h_avg"])
    
    return df


def detect_meter_anomalies(df, threshold_config=None):
    """
    Detect anomalies in individual meter readings
    """
    if threshold_config is None:
        threshold_config = {
            "zero_consumption_hours": 6,  # Hours of zero consumption to flag
            "deviation_threshold": 50,    # % deviation from expected
            "voltage_low": 200,           # Low voltage threshold
            "voltage_high": 250,          # High voltage threshold
            "pf_low": 0.7,               # Low power factor threshold
            "sudden_drop_threshold": -0.8  # 80% sudden drop
        }
    
    anomalies = []
    
    for meter_id, meter_data in df.groupby("meter_id"):
        # Zero consumption anomaly
        zero_readings = meter_data[meter_data["is_zero_consumption"] == 1]
        if len(zero_readings) >= threshold_config["zero_consumption_hours"]:
            anomalies.append({
                "meter_id": meter_id,
                "anomaly_type": "zero_consumption",
                "severity": "high" if len(zero_readings) >= 12 else "medium",
                "count": len(zero_readings),
                "latest_timestamp": zero_readings["timestamp"].max().isoformat()
            })
        
        # Consumption deviation anomaly
        if "consumption_deviation_pct" in meter_data.columns:
            high_deviation = meter_data[
                meter_data["consumption_deviation_pct"] > threshold_config["deviation_threshold"]
            ]
            if len(high_deviation) > 0:
                anomalies.append({
                    "meter_id": meter_id,
                    "anomaly_type": "consumption_deviation",
                    "severity": "high" if meter_data["consumption_deviation_pct"].max() > 80 else "medium",
                    "max_deviation": meter_data["consumption_deviation_pct"].max(),
                    "count": len(high_deviation),
                    "latest_timestamp": high_deviation["timestamp"].max().isoformat()
                })
        
        # Voltage anomaly
        if "voltage_anomaly" in meter_data.columns:
            voltage_issues = meter_data[meter_data["voltage_anomaly"] == 1]
            if len(voltage_issues) > 0:
                anomalies.append({
                    "meter_id": meter_id,
                    "anomaly_type": "voltage_anomaly",
                    "severity": "medium",
                    "count": len(voltage_issues),
                    "latest_timestamp": voltage_issues["timestamp"].max().isoformat()
                })
        
        # Sudden consumption drop
        sudden_drops = meter_data[
            meter_data["consumption_change_24h"] < threshold_config["sudden_drop_threshold"]
        ]
        if len(sudden_drops) > 0:
            anomalies.append({
                "meter_id": meter_id,
                "anomaly_type": "sudden_consumption_drop",
                "severity": "high",
                "max_drop": meter_data["consumption_change_24h"].min(),
                "count": len(sudden_drops),
                "latest_timestamp": sudden_drops["timestamp"].max().isoformat()
            })
    
    return anomalies


def calculate_theft_probability(df, anomalies_list):
    """
    Calculate theft probability for each meter based on anomalies
    """
    theft_scores = {}
    
    for meter_id in df["meter_id"].unique():
        meter_anomalies = [a for a in anomalies_list if a["meter_id"] == meter_id]
        
        if not meter_anomalies:
            theft_scores[meter_id] = 0.0
            continue
        
        base_score = 0.0
        
        for anomaly in meter_anomalies:
            if anomaly["anomaly_type"] == "zero_consumption":
                base_score += 0.4 if anomaly["severity"] == "high" else 0.2
            elif anomaly["anomaly_type"] == "consumption_deviation":
                base_score += 0.3 if anomaly["severity"] == "high" else 0.15
            elif anomaly["anomaly_type"] == "voltage_anomaly":
                base_score += 0.1
            elif anomaly["anomaly_type"] == "sudden_consumption_drop":
                base_score += 0.35
        
        # Cap at 1.0
        theft_scores[meter_id] = min(base_score, 1.0)
    
    return theft_scores
