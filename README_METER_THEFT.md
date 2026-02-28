# Individual Power Meter Theft Detection

This enhancement adds individual power meter tracking to the existing transformer-level theft detection system, allowing you to identify which specific power meters are stealing power.

## New Features

### 1. Individual Meter Tracking
- **PowerMeterReading Model**: Tracks consumption, voltage, current, and power factor for each meter
- **MeterTheftAlert Model**: Manages theft alerts with severity levels and resolution tracking
- **Real-time Anomaly Detection**: Identifies zero consumption, unusual patterns, and electrical parameter anomalies

### 2. Advanced Detection Features
- **Zero Consumption Detection**: Flags meters with unusually low or zero readings
- **Consumption Deviation Analysis**: Compares actual vs expected consumption
- **Electrical Parameter Monitoring**: Detects voltage/current anomalies and power factor issues
- **Pattern Recognition**: Identifies unusual consumption patterns during off-peak hours
- **Time-based Analysis**: Tracks consumption stability and sudden drops

### 3. Machine Learning Integration
- **Meter-level ML Model**: Random Forest classifier trained on individual meter data
- **Feature Engineering**: 15+ specialized features for meter-level theft detection
- **Fallback Rule-based System**: Ensures detection even when ML model is unavailable

## API Endpoints

### Meter Management
- `GET /api/meters/` - List all meters with latest readings and alert counts
- `GET /api/meters/readings/?meter_id=METER_001` - Get historical readings for a specific meter

### Theft Detection
- `POST /api/meters/detect/` - Real-time theft detection for meter readings
  ```json
  {
    "meter_id": "METER_001",
    "energy_consumed_kwh": 45.2,
    "expected_consumption_kwh": 50.0,
    "voltage_v": 230.5,
    "current_a": 20.3,
    "power_factor": 0.92,
    "timestamp": "2024-01-15T10:30:00",
    "reading_source": "automated"
  }
  ```

### Alert Management
- `GET /api/meters/alerts/` - Get theft alerts with filtering options
- `POST /api/meters/alerts/{alert_id}/resolve/` - Resolve theft alerts

## Setup Instructions

### 1. Database Migration
```bash
python manage.py makemigrations monitoring
python manage.py migrate
```

### 2. Populate Sample Data (Optional)
```bash
python manage.py shell < monitoring/populate_meters.py
```

### 3. Train Meter-level Model
```bash
python monitoring/meter_train.py
```

### 4. Start the Development Server
```bash
python manage.py runserver
```

## Detection Logic

### Rule-based Detection
The system uses multiple detection rules:

1. **Zero Consumption**: Consumption ≤ 0.01 kWh → 80% theft probability
2. **High Deviation**: >50% deviation from expected → High theft probability
3. **Voltage Anomalies**: Voltage < 200V or > 250V → Medium risk
4. **Current Anomalies**: Low current with consumption → High theft probability

### Machine Learning Features
- Time-based features (hour, weekday, peak hours)
- Consumption patterns (24h/7d averages, stability)
- Deviation metrics (percentage, rolling averages)
- Electrical parameters (voltage, current, power factor)
- Anomaly flags (zero consumption, unusual hours)

## Alert Severity Levels

- **Critical**: Theft probability > 90%
- **High**: Theft probability 70-90%
- **Medium**: Theft probability 40-70%
- **Low**: Theft probability < 40%

## Usage Examples

### Detect Theft for a Meter Reading
```bash
curl -X POST http://localhost:8000/api/meters/detect/ \
  -H "Content-Type: application/json" \
  -d '{
    "meter_id": "METER_001",
    "energy_consumed_kwh": 0.0,
    "expected_consumption_kwh": 45.5,
    "timestamp": "2024-01-15T14:30:00"
  }'
```

### Get Unresolved Alerts
```bash
curl "http://localhost:8000/api/meters/alerts/?is_resolved=false&severity=high"
```

### Get Meter History
```bash
curl "http://localhost:8000/api/meters/readings/?meter_id=METER_001&start_date=2024-01-10&limit=100"
```

## Model Performance

The meter-level model includes:
- **Feature Importance Tracking**: Identifies most predictive features
- **Classification Metrics**: Precision, recall, F1-score
- **Confusion Matrix**: Detailed performance analysis
- **Cross-validation**: Ensures model generalization

## Integration with Existing System

The meter-level system integrates seamlessly with the existing transformer-level detection:
- Shares the same database and Django settings
- Complements transformer-level alerts with meter-specific details
- Uses consistent API patterns and response formats
- Maintains backward compatibility with existing endpoints

## Data Quality

The system includes data quality checks:
- **Voltage Validation**: Flags unrealistic voltage readings
- **Current Validation**: Detects inconsistent current/consumption ratios
- **Power Factor Monitoring**: Identifies unusual power factor values
- **Reading Source Tracking**: Distinguishes between manual and automated readings

## Future Enhancements

Potential improvements:
- Real-time streaming data processing
- Advanced time series analysis
- Geospatial pattern detection
- Integration with SCADA systems
- Mobile alert notifications
- Automated meter reading integration
