def create_features(df):
    df = df.sort_values(["transformer_id", "timestamp"])

    # ---- Time features ----
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_peak_hour"] = df["hour"].between(18, 22).astype(int)

    # ---- Rolling loss patterns ----
    df["loss_ratio_24h_avg"] = (
        df.groupby("transformer_id")["loss_ratio"]
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["loss_ratio_7d_avg"] = (
        df.groupby("transformer_id")["loss_ratio"]
        .rolling(168)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["supply_24h_avg"] = (
        df.groupby("transformer_id")["energy_supplied_kwh"]
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df = df.dropna()

    return df
