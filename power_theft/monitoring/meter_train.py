import os
import django
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import our modules
from .models import CashPowerMeter, PowerMeterReading, MeterTheftAlert
from .meter_features import create_meter_features, detect_meter_anomalies, calculate_theft_probability
from .meter_models_ml import meter_random_forest_model, METER_FEATURES_BASIC, METER_FEATURES_ADVANCED

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "power_theft.settings")
django.setup()

def generate_synthetic_meter_data(num_meters=50, days=30):
    """
    Generate synthetic meter data for training and testing
    """
    print(f"Generating synthetic data for {num_meters} meters over {days} days...")
    
    meters = []
    readings = []
    
    # Create meters
    for i in range(num_meters):
        meter_id = f"METER_{i+1:03d}"
        
        # Randomly assign some meters as "thieves" (20%)
        is_thief = np.random.random() < 0.2
        
        meter = CashPowerMeter(
            meter_id=meter_id,
            transformer_id=f"TRANS_{(i//5)+1:03d}",  # 5 meters per transformer
            installation_date=datetime.now() - timedelta(days=365),
            is_active=True
        )
        meters.append(meter)
        
        # Generate readings for each meter
        base_consumption = np.random.uniform(50, 200)  # Base daily consumption in kWh
        
        for day in range(days):
            for hour in range(24):
                timestamp = datetime.now() - timedelta(days=days-day, hours=hour)
                
                # Generate consumption pattern
                if is_thief:
                    # Thieves have reduced consumption
                    if np.random.random() < 0.3:  # 30% chance of zero consumption
                        consumption = 0
                    else:
                        consumption = base_consumption / 24 * np.random.uniform(0.3, 0.7)
                else:
                    # Normal consumption pattern
                    hour_factor = 1.0
                    if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak hours
                        hour_factor = np.random.uniform(1.2, 1.5)
                    elif 0 <= hour <= 6:  # Off-peak
                        hour_factor = np.random.uniform(0.3, 0.6)
                    
                    consumption = (base_consumption / 24) * hour_factor * np.random.uniform(0.8, 1.2)
                
                # Add some electrical parameters
                voltage = np.random.normal(230, 5)  # Normal voltage around 230V
                current = consumption * 1000 / (voltage * 0.9) if consumption > 0 else 0.1  # Approximate current
                power_factor = np.random.uniform(0.85, 0.95) if consumption > 0 else 0.1
                
                reading = {
                    'meter_id': meter_id,
                    'timestamp': timestamp,
                    'energy_consumed_kwh': max(0, consumption),
                    'expected_consumption_kwh': base_consumption / 24,  # Expected average
                    'voltage_v': voltage,
                    'current_a': current,
                    'power_factor': power_factor,
                    'is_thief': is_thief  # Ground truth for training
                }
                readings.append(reading)
    
    return meters, readings

def prepare_meter_training_data():
    """
    Prepare training data for meter-level theft detection
    """
    print("Preparing meter training data...")
    
    # Check if we have real data, otherwise generate synthetic
    if PowerMeterReading.objects.count() < 1000:
        print("Insufficient real data, generating synthetic data...")
        meters, readings_data = generate_synthetic_meter_data()
        
        # Save meters to database
        for meter in meters:
            # Get or create transformer
            from .models import Transformer
            transformer, _ = Transformer.objects.get_or_create(
                transformer_id=meter.transformer_id,
                defaults={'capacity_kva': 100.0}
            )
            
            # Create meter
            meter_obj, _ = CashPowerMeter.objects.get_or_create(
                meter_id=meter.meter_id,
                defaults={
                    'transformer': transformer,
                    'installation_date': meter.installation_date,
                    'is_active': meter.is_active
                }
            )
            
            # Create readings
            for reading in readings_data:
                if reading['meter_id'] == meter.meter_id:
                    PowerMeterReading.objects.create(
                        meter=meter_obj,
                        timestamp=reading['timestamp'],
                        energy_consumed_kwh=reading['energy_consumed_kwh'],
                        expected_consumption_kwh=reading['expected_consumption_kwh'],
                        voltage_v=reading['voltage_v'],
                        current_a=reading['current_a'],
                        power_factor=reading['power_factor']
                    )
    
    # Load data from database
    readings = PowerMeterReading.objects.all().values(
        'meter_id', 'timestamp', 'energy_consumed_kwh', 'expected_consumption_kwh',
        'voltage_v', 'current_a', 'power_factor'
    )
    
    df = pd.DataFrame(list(readings))
    
    if df.empty:
        raise ValueError("No meter readings found in database")
    
    # Create features
    df = create_meter_features(df)
    
    # Create labels (for demonstration, using anomaly detection)
    # In real scenario, you'd have labeled theft data
    anomalies = detect_meter_anomalies(df)
    theft_scores = calculate_theft_probability(df, anomalies)
    
    # Create binary labels based on theft probability
    df['theft_label'] = df['meter_id'].map(theft_scores).apply(lambda x: 1 if x > 0.5 else 0)
    
    print(f"Training data prepared: {len(df)} samples, {df['theft_label'].sum()} theft cases")
    return df

def train_meter_model():
    """
    Train the meter-level theft detection model
    """
    print("Training meter-level theft detection model...")
    
    # Prepare data
    df = prepare_meter_training_data()
    
    # Select features
    features = [f for f in METER_FEATURES_ADVANCED if f in df.columns]
    print(f"Using {len(features)} features: {features}")
    
    X = df[features]
    y = df['theft_label']
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = meter_random_forest_model()
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    # Save model and scaler
    model_path = os.path.join(settings.BASE_DIR, 'meter_model.pkl')
    scaler_path = os.path.join(settings.BASE_DIR, 'meter_scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    return model, scaler, features

def predict_meter_theft(meter_data):
    """
    Predict theft for individual meter readings
    """
    try:
        # Load trained model
        model_path = os.path.join(settings.BASE_DIR, 'meter_model.pkl')
        scaler_path = os.path.join(settings.BASE_DIR, 'meter_scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Trained model not found. Please train the model first.")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Prepare data
        if isinstance(meter_data, dict):
            df = pd.DataFrame([meter_data])
        else:
            df = pd.DataFrame(meter_data)
        
        # Create features
        df = create_meter_features(df)
        
        # Select features
        features = [f for f in METER_FEATURES_ADVANCED if f in df.columns]
        X = df[features].fillna(df[features].mean())
        
        # Make predictions
        X_scaled = scaler.transform(X)
        theft_probabilities = model.predict_proba(X_scaled)[:, 1]
        theft_predictions = model.predict(X_scaled)
        
        results = []
        for i, row in df.iterrows():
            result = {
                'meter_id': row['meter_id'],
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                'theft_probability': round(theft_probabilities[i], 3),
                'theft_detected': bool(theft_predictions[i]),
                'energy_consumed_kwh': row['energy_consumed_kwh']
            }
            
            # Add additional info if available
            if 'consumption_deviation_pct' in row:
                result['consumption_deviation_pct'] = round(row['consumption_deviation_pct'], 2)
            
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error in meter theft prediction: {str(e)}")
        raise

if __name__ == "__main__":
    train_meter_model()
