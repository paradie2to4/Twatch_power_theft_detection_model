import os
import django
import pandas as pd
import random
from datetime import datetime, timedelta

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "power_theft.settings")
django.setup()

from monitoring.models import Transformer, LocationTheftIncident, TheftHotspot

# Rwanda location data for realistic testing
RWANDA_LOCATIONS = {
    'Kigali': {
        'districts': ['Gasabo', 'Kicukiro', 'Nyarugenge'],
        'coordinates': {'lat': -1.9441, 'lng': 30.0619},
        'transformers': 15
    },
    'Northern Province': {
        'districts': ['Burera', 'Gicumbi', 'Gakenke', 'Musanze', 'Rulindo'],
        'coordinates': {'lat': -1.5, 'lng': 29.8},
        'transformers': 8
    },
    'Southern Province': {
        'districts': ['Gisagara', 'Huye', 'Kamonyi', 'Muhanga', 'Nyamagabe', 'Nyanza', 'Nyaruguru', 'Ruhango'],
        'coordinates': {'lat': -2.3, 'lng': 29.6},
        'transformers': 10
    },
    'Eastern Province': {
        'districts': ['Bugesera', 'Gatsibo', 'Kayonza', 'Kirehe', 'Ngoma', 'Nyagatare', 'Rwamagana'],
        'coordinates': {'lat': -1.6, 'lng': 30.4},
        'transformers': 12
    },
    'Western Province': {
        'districts': ['Karongi', 'Ngororero', 'Nyabihu', 'Nyamasheke', 'Rubavu', 'Rusizi', 'Rutsiro'],
        'coordinates': {'lat': -2.0, 'lng': 29.3},
        'transformers': 9
    }
}

def generate_realistic_coordinates(base_lat, base_lng, radius_km=5):
    """Generate realistic coordinates within a radius"""
    import math
    
    # Convert radius from km to degrees (approximate)
    lat_offset = radius_km / 111.0
    lng_offset = radius_km / (111.0 * math.cos(math.radians(base_lat)))
    
    lat = base_lat + random.uniform(-lat_offset, lat_offset)
    lng = base_lng + random.uniform(-lng_offset, lng_offset)
    
    return lat, lng

def create_transformers_with_locations():
    """Create transformers with realistic Rwanda locations"""
    print("Creating transformers with location data...")
    
    transformer_count = 0
    
    for province_name, province_data in RWANDA_LOCATIONS.items():
        base_lat = province_data['coordinates']['lat']
        base_lng = province_data['coordinates']['lng']
        
        for district in province_data['districts']:
            # Generate district center coordinates
            district_lat, district_lng = generate_realistic_coordinates(base_lat, base_lng, 20)
            
            # Create transformers in this district
            num_transformers = province_data['transformers'] // len(province_data['districts'])
            if district == list(province_data['districts'])[-1]:  # Last district gets remaining
                num_transformers = province_data['transformers'] - (num_transformers * (len(province_data['districts']) - 1))
            
            for i in range(num_transformers):
                transformer_count += 1
                
                # Generate transformer location
                lat, lng = generate_realistic_coordinates(district_lat, district_lng, 10)
                
                # Create realistic address
                street_number = random.randint(1, 999)
                street_name = random.choice(['Avenue', 'Street', 'Road'])
                neighborhood = random.choice(['Sector A', 'Sector B', 'Sector C', 'Central', 'North', 'South'])
                
                address = f"{street_number} {street_name}, {neighborhood}, {district}"
                
                transformer = Transformer.objects.create(
                    transformer_id=f"TRANS_{transformer_count:03d}",
                    location=f"{district}, {province_name}",
                    latitude=lat,
                    longitude=lng,
                    address=address,
                    district=district,
                    province=province_name,
                    postal_code=f"{random.randint(100, 999)}",
                    capacity_kva=random.uniform(50, 200),
                    installation_date=datetime.now() - timedelta(days=random.randint(365, 1825)),
                    transformer_type=random.choice(['pole-mounted', 'pad-mounted', 'substation']),
                    manufacturer=random.choice(['ABB', 'Siemens', 'Schneider Electric', 'GE']),
                    serial_number=f"SN{random.randint(100000, 999999)}",
                    status='active'
                )
                
                print(f"Created transformer: {transformer.transformer_id} at {transformer.full_location}")
    
    print(f"Created {transformer_count} transformers with location data")
    return transformer_count

def create_theft_hotspots():
    """Create theft hotspots based on incident data"""
    print("Creating theft hotspots...")
    
    # Group transformers by district and create hotspots
    districts = Transformer.objects.values('district', 'province').distinct()
    
    for district_data in districts:
        district = district_data['district']
        province = district_data['province']
        
        if not district or not province:
            continue
        
        # Get transformers in this district
        transformers_in_district = Transformer.objects.filter(
            district=district, 
            province=province
        )
        
        if transformers_in_district.count() == 0:
            continue
        
        # Calculate center point
        avg_lat = sum(t.latitude for t in transformers_in_district if t.latitude) / len([t for t in transformers_in_district if t.latitude])
        avg_lng = sum(t.longitude for t in transformers_in_district if t.longitude) / len([t for t in transformers_in_district if t.longitude])
        
        # Generate random incident counts for demo
        total_incidents = random.randint(0, 15)
        incidents_30_days = random.randint(0, min(5, total_incidents))
        incidents_90_days = random.randint(incidents_30_days, min(10, total_incidents))
        
        # Calculate estimated losses
        avg_loss_per_incident = random.uniform(50, 500)
        total_loss = total_incidents * avg_loss_per_incident
        
        hotspot, created = TheftHotspot.objects.get_or_create(
            district=district,
            province=province,
            area_name=f"{district} Area",
            defaults={
                'center_latitude': avg_lat,
                'center_longitude': avg_lng,
                'radius_km': random.uniform(5, 15),
                'total_incidents': total_incidents,
                'incidents_last_30_days': incidents_30_days,
                'incidents_last_90_days': incidents_90_days,
                'total_estimated_loss_usd': total_loss,
                'average_loss_per_incident': avg_loss_per_incident
            }
        )
        
        if created:
            hotspot.update_risk_level()
            print(f"Created hotspot: {hotspot.area_name} - {hotspot.risk_level} risk")
        else:
            print(f"Hotspot already exists: {hotspot.area_name}")

def create_sample_theft_incidents():
    """Create sample theft incidents with location data"""
    print("Creating sample theft incidents...")
    
    # Get transformers with location data
    transformers = Transformer.objects.filter(
        latitude__isnull=False,
        longitude__isnull=False
    )
    
    if transformers.count() == 0:
        print("No transformers with location data found. Run create_transformers_with_locations() first.")
        return
    
    incident_count = 0
    
    for transformer in transformers:
        # Randomly decide if this transformer has incidents (30% chance)
        if random.random() < 0.3:
            # Generate 1-3 incidents per transformer
            num_incidents = random.randint(1, 3)
            
            for i in range(num_incidents):
                incident_count += 1
                
                # Generate incident date (last 6 months)
                days_ago = random.randint(1, 180)
                incident_date = datetime.now() - timedelta(days=days_ago)
                
                # Random theft type
                theft_type = random.choice([
                    'energy_theft', 'meter_tampering', 'bypass', 'illegal_connection'
                ])
                
                # Random detection method
                detection_method = random.choice([
                    'behavioral_analysis', 'manual_inspection', 'customer_complaint', 'routine_audit'
                ])
                
                # Calculate estimated losses
                estimated_loss_kwh = random.uniform(100, 2000)
                estimated_loss_usd = estimated_loss_kwh * random.uniform(0.08, 0.15)  # $0.08-0.15 per kWh
                
                incident = LocationTheftIncident.objects.create(
                    transformer=transformer,
                    incident_date=incident_date,
                    theft_type=theft_type,
                    detection_method=detection_method,
                    confidence_score=random.uniform(0.6, 0.95),
                    estimated_loss_kwh=estimated_loss_kwh,
                    estimated_loss_usd=estimated_loss_usd,
                    status=random.choice(['reported', 'investigating', 'confirmed', 'resolved']),
                    
                    # Copy location from transformer
                    latitude=transformer.latitude,
                    longitude=transformer.longitude,
                    address=transformer.address,
                    district=transformer.district,
                    province=transformer.province
                )
                
                print(f"Created incident: {incident.id} at {incident.location_summary}")
    
    print(f"Created {incident_count} theft incidents")
    return incident_count

def main():
    """Main function to populate location data"""
    print("Populating location data for Rwanda power grid...")
    
    # Create transformers with locations
    transformer_count = create_transformers_with_locations()
    
    # Create theft incidents
    incident_count = create_sample_theft_incidents()
    
    # Create hotspots
    create_theft_hotspots()
    
    print(f"\nSummary:")
    print(f"- Created {transformer_count} transformers with Rwanda locations")
    print(f"- Created {incident_count} theft incidents")
    print(f"- Created theft hotspots for high-risk areas")
    print(f"- Ready for location-based theft detection and mapping!")

if __name__ == "__main__":
    main()
