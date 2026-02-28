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
from sklearn.ensemble import IsolationForest

# Import our modules
from .models import Transformer, TransformerReading, TransformerBehaviorAlert
from .transformer_behavior_features import (
    create_transformer_behavior_features,
    detect_transformer_behavioral_anomalies,
    TRANSFORMER_BEHAVIOR_FEATURES_ADVANCED,
    TRANSFORMER_BEHAVIOR_FEATURES_BASIC
)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "power_theft.settings")
django.setup()

def generate_synthetic_transformer_behavior_data(num_transformers=20, days=30):
    """
    Generate synthetic transformer data with behavioral anomalies for training
    """
    print(f"Generating synthetic transformer behavior data for {num_transformers} transformers over {days} days...")
    
    readings = []
    
    for i in range(num_transformers):
        transformer_id = f"TRANS_{i+1:03d}"
        capacity_kva = np.random.uniform(50, 200)
        
        # Create baseline behavior pattern for this transformer
        base_current = np.random.uniform(50, 150)  # Base current in Amperes
        base_voltage = 230  # Standard voltage
        
        # Randomly assign some transformers to have anomalies
        has_anomalies = np.random.random() < 0.3  # 30% have anomalies
        anomaly_type = np.random.choice(['load_spike', 'off_peak_load', 'instability']) if has_anomalies else None
        
        for day in range(days):
            for hour in range(24):
                timestamp = datetime.now() - timedelta(days=days-day, hours=hour)
                
                # Generate normal behavior pattern
                hour_factor = 1.0
                if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak hours
                    hour_factor = np.random.uniform(1.2, 1.5)
                elif 0 <= hour <= 6:  # Off-peak
                    hour_factor = np.random.uniform(0.3, 0.6)
                
                current = base_current * hour_factor * np.random.uniform(0.9, 1.1)
                voltage = base_voltage + np.random.normal(0, 2)  # Small voltage variations
                
                # Inject anomalies
                if has_anomalies and anomaly_type:
                    if anomaly_type == 'load_spike' and np.random.random() < 0.05:  # 5% chance
                        current *= np.random.uniform(2.0, 3.0)  # Sudden load spike
                    elif anomaly_type == 'off_peak_load' and hour in range(0, 6) and np.random.random() < 0.1:
                        current *= np.random.uniform(1.5, 2.0)  # Unusual off-peak load
                    elif anomaly_type == 'instability' and np.random.random() < 0.15:
                        current += np.random.normal(0, base_current * 0.3)  # High variance
                        voltage += np.random.normal(0, 5)
                
                # Calculate derived values
                power_factor = np.random.uniform(0.85, 0.95)
                frequency = 50 + np.random.normal(0, 0.1)
                
                # Energy calculations (simplified)
                energy_supplied = (current * voltage * power_factor * np.sqrt(3)) / 1000  # kW
                energy_consumed = energy_supplied * np.random.uniform(0.7, 0.95)  # Some losses
                
                reading = {
                    'transformer_id': transformer_id,
                    'capacity_kva': capacity_kva,
                    'timestamp': timestamp,
                    'energy_supplied_kwh': energy_supplied,
                    'total_meter_consumption_kwh': energy_consumed,
                    'output_current_a': current,
                    'output_voltage_v': voltage,
                    'power_factor': power_factor,
                    'frequency_hz': frequency,
                    'loss_kwh': energy_supplied - energy_consumed,
                    'loss_ratio': (energy_supplied - energy_consumed) / energy_supplied if energy_supplied > 0 else 0,
                    'has_anomaly': has_anomalies,
                    'anomaly_type': anomaly_type if has_anomalies else 'normal'
                }
                readings.append(reading)
    
    return readings

def prepare_transformer_behavior_training_data():
    """
    Prepare training data for transformer behavior model
    """
    print("Preparing transformer behavior training data...")
    
    # Check if we have real data, otherwise generate synthetic
    if TransformerReading.objects.count() < 1000:
        print("Insufficient real data, generating synthetic data...")
        readings_data = generate_synthetic_transformer_behavior_data()
        
        # Save to database
        for reading in readings_data:
            # Get or create transformer
            transformer, _ = Transformer.objects.get_or_create(
                transformer_id=reading['transformer_id'],
                defaults={'capacity_kva': reading['capacity_kva']}
            )
            
            # Create reading
            TransformerReading.objects.create(
                transformer=transformer,
                timestamp=reading['timestamp'],
                energy_supplied_kwh=reading['energy_supplied_kwh'],
                total_meter_consumption_kwh=reading['total_meter_consumption_kwh'],
                output_current_a=reading['output_current_a'],
                output_voltage_v=reading['output_voltage_v'],
                power_factor=reading['power_factor'],
                frequency_hz=reading['frequency_hz'],
                loss_kwh=reading['loss_kwh'],
                loss_ratio=reading['loss_ratio'],
                theft_flag=reading['has_anomaly']
            )
    
    # Load data from database
    readings = TransformerReading.objects.select_related('transformer').all()
    
    data = []
    for reading in readings:
        data.append({
            'transformer_id': reading.transformer.transformer_id,
            'capacity_kva': reading.transformer.capacity_kva,
            'timestamp': reading.timestamp,
            'energy_supplied_kwh': reading.energy_supplied_kwh,
            'total_meter_consumption_kwh': reading.total_meter_consumption_kwh,
            'output_current_a': reading.output_current_a,
            'output_voltage_v': reading.output_voltage_v,
            'power_factor': reading.power_factor,
            'frequency_hz': reading.frequency_hz,
            'loss_kwh': reading.loss_kwh,
            'loss_ratio': reading.loss_ratio,
            'theft_flag': reading.theft_flag
        })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        raise ValueError("No transformer readings found in database")
    
    # Create behavioral features
    df = detect_transformer_behavioral_anomalies(df)
    
    print(f"Training data prepared: {len(df)} samples")
    return df

def train_transformer_behavior_model():
    """
    Train the transformer behavior baseline model
    """
    print("Training Transformer Behavior Baseline Model...")
    
    # Prepare data
    df = prepare_transformer_behavior_training_data()
    
    # Select features
    features = [f for f in TRANSFORMER_BEHAVIOR_FEATURES_ADVANCED if f in df.columns]
    print(f"Using {len(features)} features: {features}")
    
    # Create labels for anomaly detection
    # Multi-class classification: normal, load_spike, off_peak_load, instability
    y = df['anomaly_type']
    
    X = df[features]
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importance:")
    print(feature_importance.head(10))
    
    # Save model and scaler
    model_path = os.path.join(settings.BASE_DIR, 'transformer_behavior_model.pkl')
    scaler_path = os.path.join(settings.BASE_DIR, 'transformer_behavior_scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    # Save some test predictions to database
    save_behavior_predictions_to_db(X_test, y_test, y_pred, y_prob, df.loc[X_test.index])
    
    return model, scaler, features

def save_behavior_predictions_to_db(X_test, y_test, y_pred, y_prob, test_metadata):
    """
    Save behavior predictions to TransformerBehaviorAlert table
    """
    print("Saving behavior predictions to database...")
    
    records = []
    for i, idx in enumerate(X_test.index):
        row = test_metadata.iloc[idx]
        
        # Get the predicted class and probability
        predicted_class = y_pred[i]
        class_probabilities = y_prob[i]
        confidence = np.max(class_probabilities)
        
        # Calculate anomaly score (0-100)
        if predicted_class == 'normal':
            anomaly_score = np.random.uniform(0, 20)  # Low scores for normal
        else:
            anomaly_score = np.random.uniform(60, 95)  # High scores for anomalies
        
        try:
            transformer = Transformer.objects.get(transformer_id=row['transformer_id'])
            
            record = TransformerBehaviorAlert(
                transformer=transformer,
                timestamp=row['timestamp'],
                anomaly_score=anomaly_score,
                anomaly_type=predicted_class,
                confidence_level=confidence,
                current_deviation=row.get('current_deviation_pct', 0),
                voltage_deviation=row.get('voltage_deviation_pct', 0),
                behavioral_change_magnitude=row.get('behavioral_change_magnitude', 0)
            )
            records.append(record)
            
        except Transformer.DoesNotExist:
            # Create transformer if it doesn't exist
            transformer = Transformer.objects.create(
                transformer_id=row['transformer_id'],
                capacity_kva=row.get('capacity_kva', 100.0)
            )
            
            record = TransformerBehaviorAlert(
                transformer=transformer,
                timestamp=row['timestamp'],
                anomaly_score=anomaly_score,
                anomaly_type=predicted_class,
                confidence_level=confidence,
                current_deviation=row.get('current_deviation_pct', 0),
                voltage_deviation=row.get('voltage_deviation_pct', 0),
                behavioral_change_magnitude=row.get('behavioral_change_magnitude', 0)
            )
            records.append(record)
    
    if records:
        TransformerBehaviorAlert.objects.bulk_create(records, batch_size=1000)
        print(f"Saved {len(records)} behavior predictions to database")

def predict_transformer_behavior(transformer_data):
    """
    Predict behavioral anomalies for transformer data
    """
    try:
        # Load trained model
        model_path = os.path.join(settings.BASE_DIR, 'transformer_behavior_model.pkl')
        scaler_path = os.path.join(settings.BASE_DIR, 'transformer_behavior_scaler.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Trained transformer behavior model not found. Please train the model first.")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Prepare data
        if isinstance(transformer_data, dict):
            df = pd.DataFrame([transformer_data])
        else:
            df = pd.DataFrame(transformer_data)
        
        # Create features
        df = detect_transformer_behavioral_anomalies(df)
        
        # Select features
        features = [f for f in TRANSFORMER_BEHAVIOR_FEATURES_ADVANCED if f in df.columns]
        X = df[features].fillna(df[features].mean())
        
        # Make predictions
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        results = []
        for i, row in df.iterrows():
            result = {
                'transformer_id': row['transformer_id'],
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                'anomaly_type': predictions[i],
                'anomaly_score': row['anomaly_score'],
                'confidence_level': np.max(probabilities[i]),
                'behavioral_change_magnitude': row['behavioral_change_magnitude'],
                'current_deviation_pct': row.get('current_deviation_pct', 0),
                'voltage_deviation_pct': row.get('voltage_deviation_pct', 0)
            }
            results.append(result)
        
        return results
        
    except Exception as e:
        print(f"Error in transformer behavior prediction: {str(e)}")
        raise

if __name__ == "__main__":
    train_transformer_behavior_model()
