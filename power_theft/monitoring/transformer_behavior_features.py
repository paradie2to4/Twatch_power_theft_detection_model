import pandas as pd
import numpy as np

def create_transformer_behavior_features(df):
    """
    Create features for Transformer Behavior Baseline Model
    Focuses on behavioral change detection rather than absolute thresholds
    """
    df = df.sort_values(["transformer_id", "timestamp"])
    
    # ---- Time features ----
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["is_peak_hour"] = df["hour"].between(18, 22).astype(int)
    df["is_off_peak"] = df["hour"].between(0, 6).astype(int)
    
    # ---- Historical averages (defines "normal") ----
    # Per-transformer historical baselines
    df["current_24h_avg"] = (
        df.groupby("transformer_id")["output_current_a"]
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    df["current_7d_avg"] = (
        df.groupby("transformer_id")["output_current_a"]
        .rolling(168)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    df["voltage_24h_avg"] = (
        df.groupby("transformer_id")["output_voltage_v"]
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    df["voltage_7d_avg"] = (
        df.groupby("transformer_id")["output_voltage_v"]
        .rolling(168)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # ---- Rolling variance (detects instability) ----
    df["current_variance_24h"] = (
        df.groupby("transformer_id")["output_current_a"]
        .rolling(24)
        .var()
        .reset_index(level=0, drop=True)
    )
    
    df["voltage_variance_24h"] = (
        df.groupby("transformer_id")["output_voltage_v"]
        .rolling(24)
        .var()
        .reset_index(level=0, drop=True)
    )
    
    # ---- Rate of change (detects sudden events) ----
    df["current_rate_of_change"] = (
        df.groupby("transformer_id")["output_current_a"]
        .diff()
        .fillna(0)
    )
    
    df["voltage_rate_of_change"] = (
        df.groupby("transformer_id")["output_voltage_v"]
        .diff()
        .fillna(0)
    )
    
    # ---- Deviation from normal (key for behavioral change) ----
    df["current_deviation_pct"] = (
        (df["output_current_a"] - df["current_24h_avg"]) / df["current_24h_avg"].replace(0, np.nan)
    ).fillna(0) * 100
    
    df["voltage_deviation_pct"] = (
        (df["output_voltage_v"] - df["voltage_24h_avg"]) / df["voltage_24h_avg"].replace(0, np.nan)
    ).fillna(0) * 100
    
    # ---- Contextual scaling using transformer rating ----
    if "capacity_kva" in df.columns:
        # Convert to per-unit values for comparison across transformers
        df["current_per_unit"] = df["output_current_a"] / (df["capacity_kva"] * 1000 / (df["output_voltage_v"] * np.sqrt(3)))
        df["load_percentage"] = (df["output_current_a"] * df["output_voltage_v"] * np.sqrt(3)) / (df["capacity_kva"] * 1000) * 100
    
    # ---- Behavioral pattern features ----
    # Load pattern consistency
    df["load_pattern_stability"] = (
        df.groupby("transformer_id")["current_deviation_pct"]
        .rolling(24)
        .std()
        .reset_index(level=0, drop=True)
    )
    
    # Off-peak activity detection (unusual behavior)
    df["off_peak_current_ratio"] = (
        df.groupby("transformer_id").apply(
            lambda x: x["output_current_a"] / x[x["is_off_peak"] == 0]["output_current_a"].mean()
        ).reset_index(level=0, drop=True)
    )
    
    # ---- Advanced anomaly indicators ----
    # Z-scores for statistical anomaly detection
    df["current_zscore"] = (
        (df["output_current_a"] - df["current_7d_avg"]) / df["current_variance_24h"].replace(0, np.nan).pow(0.5)
    ).fillna(0)
    
    df["voltage_zscore"] = (
        (df["output_voltage_v"] - df["voltage_7d_avg"]) / df["voltage_variance_24h"].replace(0, np.nan).pow(0.5)
    ).fillna(0)
    
    # Composite behavioral change score
    df["behavioral_change_score"] = (
        np.abs(df["current_zscore"]) * 0.4 +  # Weight current changes more
        np.abs(df["voltage_zscore"]) * 0.3 +   # Voltage changes
        np.abs(df["current_rate_of_change"]) * 0.2 +  # Sudden current changes
        df["load_pattern_stability"].fillna(0) * 0.1  # Pattern instability
    )
    
    # Remove rows with insufficient data for feature calculation
    df = df.dropna(subset=["current_24h_avg", "voltage_24h_avg"])
    
    return df


def classify_anomaly_type(row):
    """
    Classify the type of anomaly based on feature values
    """
    current_zscore = abs(row.get("current_zscore", 0))
    voltage_zscore = abs(row.get("voltage_zscore", 0))
    current_rate = abs(row.get("current_rate_of_change", 0))
    voltage_rate = abs(row.get("voltage_rate_of_change", 0))
    is_off_peak = row.get("is_off_peak", 0)
    load_stability = row.get("load_pattern_stability", 0)
    
    # Load spike detection
    if current_zscore > 2.5 and current_rate > 10:
        return "load_spike"
    
    # Off-peak unusual activity
    elif is_off_peak and current_zscore > 1.5 and row.get("off_peak_current_ratio", 0) > 0.3:
        return "off_peak_load"
    
    # Instability detection
    elif load_stability > 50 or (current_zscore > 1.5 and voltage_zscore > 1.5):
        return "instability"
    
    # Voltage-specific issues
    elif voltage_zscore > 2.0:
        return "voltage_anomaly"
    
    # Current-specific issues
    elif current_zscore > 2.0:
        return "current_anomaly"
    
    else:
        return "normal"


def calculate_anomaly_score_0_100(row):
    """
    Calculate anomaly score on 0-100 scale
    """
    current_zscore = abs(row.get("current_zscore", 0))
    voltage_zscore = abs(row.get("voltage_zscore", 0))
    current_rate = abs(row.get("current_rate_of_change", 0))
    voltage_rate = abs(row.get("voltage_rate_of_change", 0))
    load_stability = row.get("load_pattern_stability", 0)
    behavioral_change = row.get("behavioral_change_score", 0)
    
    # Base score from behavioral change
    base_score = min(behavioral_change * 10, 80)  # Cap at 80 for base score
    
    # Additional score for extreme events
    if current_zscore > 3:
        base_score += 10
    elif voltage_zscore > 3:
        base_score += 10
    elif current_rate > 20:
        base_score += 5
    elif voltage_rate > 10:
        base_score += 5
    
    # Instability penalty
    if load_stability > 100:
        base_score += 10
    
    return min(base_score, 100)


def calculate_confidence_level(row):
    """
    Calculate confidence level (0.0 to 1.0) based on data quality and pattern strength
    """
    # Data quality factors
    has_current = not pd.isna(row.get("output_current_a"))
    has_voltage = not pd.isna(row.get("output_voltage_v"))
    data_quality = (has_current + has_voltage) / 2
    
    # Pattern strength (how consistent is the deviation?)
    current_deviation = abs(row.get("current_deviation_pct", 0))
    voltage_deviation = abs(row.get("voltage_deviation_pct", 0))
    pattern_strength = min((current_deviation + voltage_deviation) / 50, 1.0)
    
    # Historical data availability
    has_24h_data = not pd.isna(row.get("current_24h_avg"))
    has_7d_data = not pd.isna(row.get("current_7d_avg"))
    data_availability = (has_24h_data + has_7d_data) / 2
    
    # Combined confidence
    confidence = (data_quality * 0.3 + pattern_strength * 0.4 + data_availability * 0.3)
    
    return min(confidence, 1.0)


def detect_transformer_behavioral_anomalies(df):
    """
    Main function to detect behavioral anomalies in transformers
    Returns DataFrame with anomaly scores, types, and confidence levels
    """
    # Create features
    df_features = create_transformer_behavior_features(df)
    
    # Apply classification and scoring
    df_features["anomaly_type"] = df_features.apply(classify_anomaly_type, axis=1)
    df_features["anomaly_score"] = df_features.apply(calculate_anomaly_score_0_100, axis=1)
    df_features["confidence_level"] = df_features.apply(calculate_confidence_level, axis=1)
    
    # Calculate behavioral change magnitude
    df_features["behavioral_change_magnitude"] = np.sqrt(
        df_features["current_deviation_pct"]**2 + df_features["voltage_deviation_pct"]**2
    )
    
    return df_features


# Feature sets for different model complexity levels
TRANSFORMER_BEHAVIOR_FEATURES_BASIC = [
    "output_current_a",
    "output_voltage_v", 
    "hour",
    "weekday",
    "current_24h_avg",
    "voltage_24h_avg",
    "current_deviation_pct",
    "voltage_deviation_pct"
]

TRANSFORMER_BEHAVIOR_FEATURES_ADVANCED = [
    "output_current_a",
    "output_voltage_v",
    "hour", 
    "weekday",
    "is_peak_hour",
    "is_off_peak",
    "current_24h_avg",
    "current_7d_avg",
    "voltage_24h_avg", 
    "voltage_7d_avg",
    "current_variance_24h",
    "voltage_variance_24h",
    "current_rate_of_change",
    "voltage_rate_of_change",
    "current_deviation_pct",
    "voltage_deviation_pct",
    "current_zscore",
    "voltage_zscore",
    "load_pattern_stability"
]

TRANSFORMER_BEHAVIOR_FEATURES_FULL = TRANSFORMER_BEHAVIOR_FEATURES_ADVANCED + [
    "capacity_kva",
    "current_per_unit", 
    "load_percentage",
    "off_peak_current_ratio",
    "behavioral_change_score"
]
