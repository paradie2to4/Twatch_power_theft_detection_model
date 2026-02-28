# Location-Based Power Theft Detection

This enhancement adds comprehensive location tracking to identify exactly where power theft occurs, enabling geographic analysis and hotspot identification.

## New Location Features

### 1. Enhanced Transformer Model
- **GPS Coordinates**: Latitude and longitude for precise mapping
- **Administrative Divisions**: District, province, postal code
- **Physical Address**: Street-level location details
- **Technical Details**: Installation date, transformer type, manufacturer

### 2. Location-Based Incident Tracking
- **LocationTheftIncident Model**: Tracks theft incidents with full location context
- **TheftHotspot Model**: Identifies high-risk geographic areas
- **Geographic Analytics**: District and province-level theft patterns

### 3. Mapping and Visualization
- **Interactive Maps**: Real-time theft incident visualization
- **Hotspot Detection**: Automatic identification of high-risk areas
- **Geographic Filtering**: Search theft by location parameters

## API Endpoints

### Location Dashboard
- `GET /api/location/dashboard/` - Comprehensive location-based statistics
  - Total transformers with location coverage
  - Incidents by district/province  
  - Active theft hotspots
  - Recent location-tagged alerts

### Transformer Location Search
- `GET /api/location/transformers/` - Find transformers by location
  - Filter by district/province
  - Bounding box search (lat/lng ranges)
  - Returns transformers with latest alerts

### Theft Map Data
- `GET /api/location/map-data/` - Data for map visualization
  - All incidents and alerts with GPS coordinates
  - Severity levels and confidence scores
  - Popup information for each point

### Incident Reporting
- `POST /api/location/report-incident/` - Report theft at specific location
  ```json
  {
    "transformer_id": "TRANS_001",
    "incident_date": "2024-01-15T10:30:00",
    "theft_type": "energy_theft",
    "detection_method": "behavioral_analysis",
    "confidence_score": 0.85,
    "estimated_loss_kwh": 500.0,
    "estimated_loss_usd": 50.0,
    "investigation_notes": "Physical inspection confirmed tampered seals"
  }
  ```

### Location Analytics
- `GET /api/location/analytics/` - Geographic theft patterns
  - District-level statistics
  - Monthly trends by location
  - Theft type distribution by area
  - Detection method effectiveness

## Data Models

### Transformer Location Fields
```python
class Transformer(models.Model):
    # Location information
    latitude = models.DecimalField(max_digits=10, decimal_places=8)
    longitude = models.DecimalField(max_digits=11, decimal_places=8)
    address = models.TextField()
    district = models.CharField(max_length=100)
    province = models.CharField(max_length=100)
    postal_code = models.CharField(max_length=20)
    
    # Technical details
    installation_date = models.DateField()
    transformer_type = models.CharField(max_length=50)
    manufacturer = models.CharField(max_length=100)
    serial_number = models.CharField(max_length=100)
    status = models.CharField(max_length=20)
```

### Theft Incident Tracking
```python
class LocationTheftIncident(models.Model):
    # Location (denormalized for fast queries)
    latitude = models.DecimalField(max_digits=10, decimal_places=8)
    longitude = models.DecimalField(max_digits=11, decimal_places=8)
    district = models.CharField(max_length=100)
    province = models.CharField(max_length=100)
    
    # Incident details
    theft_type = models.CharField(max_length=50)
    detection_method = models.CharField(max_length=50)
    confidence_score = models.FloatField()
    estimated_loss_usd = models.FloatField()
    status = models.CharField(max_length=20)
```

### Hotspot Identification
```python
class TheftHotspot(models.Model):
    # Geographic area
    district = models.CharField(max_length=100)
    province = models.CharField(max_length=100)
    center_latitude = models.DecimalField(max_digits=10, decimal_places=8)
    center_longitude = models.DecimalField(max_digits=11, decimal_places=8)
    radius_km = models.FloatField()
    
    # Risk metrics
    incidents_last_30_days = models.IntegerField()
    risk_level = models.CharField(max_length=20)
    total_estimated_loss_usd = models.FloatField()
```

## Setup Instructions

### 1. Database Migration
```bash
python manage.py makemigrations monitoring
python manage.py migrate
```

### 2. Populate Location Data
```bash
python manage.py shell < populate_location_data.py
```
This creates:
- 54 transformers across Rwanda's 5 provinces and 30 districts
- Realistic GPS coordinates for each transformer
- Sample theft incidents with location context
- Theft hotspots for high-risk areas

### 3. Start Development Server
```bash
python manage.py runserver
```

## Usage Examples

### Get Location Dashboard
```bash
curl "http://localhost:8000/api/location/dashboard/"
```

### Find Transformers in District
```bash
curl "http://localhost:8000/api/location/transformers/?district=Gasabo&province=Kigali"
```

### Get Map Data for Last 30 Days
```bash
curl "http://localhost:8000/api/location/map-data/?days=30"
```

### Report Theft Incident
```bash
curl -X POST http://localhost:8000/api/location/report-incident/ \
  -H "Content-Type: application/json" \
  -d '{
    "transformer_id": "TRANS_001",
    "incident_date": "2024-01-15T10:30:00",
    "theft_type": "energy_theft",
    "detection_method": "behavioral_analysis",
    "confidence_score": 0.85,
    "estimated_loss_kwh": 500.0,
    "estimated_loss_usd": 50.0
  }'
```

### Get District Analytics
```bash
curl "http://localhost:8000/api/location/analytics/?days=90"
```

## Geographic Analysis Features

### 1. Hotspot Detection
- **Automatic Identification**: Areas with high incident concentrations
- **Risk Level Classification**: Low, Medium, High, Critical based on recent activity
- **Dynamic Updates**: Hotspot risk levels update as new incidents occur

### 2. District-Level Statistics
- **Incident Counts**: Total theft incidents by district
- **Loss Estimates**: Financial impact by geographic area
- **Detection Method Analysis**: Effectiveness by location
- **Temporal Trends**: Monthly patterns by district

### 3. Interactive Mapping
- **Real-time Updates**: Live theft incident mapping
- **Severity Visualization**: Color-coded alerts by severity
- **Detailed Popups**: Comprehensive incident information
- **Filtering Options**: Time range, severity, theft type filters

## Rwanda-Specific Implementation

The system includes realistic Rwanda location data:
- **5 Provinces**: Kigali, Northern, Southern, Eastern, Western
- **30 Districts**: All administrative districts represented
- **Realistic Coordinates**: GPS coordinates for each area
- **Local Context**: Addresses and postal codes following Rwandan conventions

## Benefits

### 1. Precise Location Identification
- **Exact Coordinates**: Pinpoint theft location to within meters
- **Administrative Context**: District and province for resource allocation
- **Physical Address**: Street-level details for field investigations

### 2. Geographic Pattern Analysis
- **Hotspot Detection**: Identify high-risk areas automatically
- **Resource Optimization**: Deploy inspection teams to high-risk zones
- **Preventive Measures**: Target infrastructure improvements where needed

### 3. Operational Efficiency
- **Fast Location Queries**: Optimized database indexes for geographic searches
- **Map Integration**: Ready for integration with mapping platforms
- **Mobile Field Support**: Location data accessible on mobile devices

## Integration with Existing System

The location tracking seamlessly integrates with:
- **Transformer Behavior Model**: Behavioral alerts include location context
- **Meter-Level Detection**: Individual meter theft mapped to transformer locations
- **Historical Analysis**: Location-based trend analysis over time
- **Alert Management**: Geographic filtering of theft alerts

This comprehensive location system enables utilities to identify exactly where power theft occurs, analyze geographic patterns, and optimize resources for theft prevention and investigation.
