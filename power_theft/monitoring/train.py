# ml/train.py
import os
import django
from .load_data import prepare_training_data
from .features import create_features
from .models_ml import random_forest_model
from .models import Transformer, TheftPrediction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from django.conf import settings
from sklearn.metrics import classification_report

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "power_theft.settings")
django.setup()

def train_and_predict():
    df = prepare_training_data()
    df = create_features(df)

    FEATURES = [
        "energy_supplied_kwh",
        "energy_consumed_kwh",
        "loss_ratio",
        "hour",
        "weekday",
        "is_peak_hour",
        "loss_ratio_24h_avg",
        "loss_ratio_7d_avg",
        "supply_24h_avg"
    ]

    X = df[FEATURES]
    y = df["theft_flag"]

    # Split data and keep metadata
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Get corresponding metadata for test set
    meta_test = df.loc[X_test.index, ["transformer_id", "timestamp", "loss_ratio"]]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = random_forest_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print(classification_report(y_test, y_pred))

    # Save predictions to DB
    records = []
    for i in range(len(meta_test)):
        transformer_id = str(meta_test.iloc[i]["transformer_id"])
        try:
            transformer = Transformer.objects.get(transformer_id=transformer_id)
            records.append(
                TheftPrediction(
                    transformer=transformer,
                    timestamp=meta_test.iloc[i]["timestamp"],
                    loss_ratio=meta_test.iloc[i]["loss_ratio"],
                    theft_probability=y_prob[i],
                    theft_detected=bool(y_pred[i]),
                    model_version="rf_v1"
                )
            )
        except Transformer.DoesNotExist:
            # Create transformer if it doesn't exist
            transformer = Transformer.objects.create(
                transformer_id=transformer_id,
                capacity_kva=100.0
            )
            records.append(
                TheftPrediction(
                    transformer=transformer,
                    timestamp=meta_test.iloc[i]["timestamp"],
                    loss_ratio=meta_test.iloc[i]["loss_ratio"],
                    theft_probability=y_prob[i],
                    theft_detected=bool(y_pred[i]),
                    model_version="rf_v1"
                )
            )
    
    if records:
        TheftPrediction.objects.bulk_create(records, batch_size=1000)
        print(f"Saved {len(records)} predictions to database")
    else:
        print("No records to save")
    
    # Save the trained model
    model_path = os.path.join(settings.BASE_DIR, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(settings.BASE_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    return model, scaler

if __name__ == "__main__":
    train_and_predict()
