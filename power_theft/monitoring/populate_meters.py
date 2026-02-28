import os
import django
import pandas as pd
from datetime import datetime, timedelta
import random

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "power_theft.settings")
django.setup()

from .models import Transformer, CashPowerMeter, PowerMeterReading

def create_sample_meters():
    """Create sample meters for testing"""
    print("Creating sample meters...")
    
    # Get existing transformers or create some
    transformers = list(Transformer.objects.all())
    if not transformers:
        # Create sample transformers
        for i in range(10):
            transformer = Transformer.objects.create(
                transformer_id=f"TRANS_{i+1:03d}",
                capacity_kva=random.uniform(50, 200),
                location=f"Location_{i+1}"
            )
            transformers.append(transformer)
    
    # Create meters for each transformer (5-10 meters per transformer)
    meter_count = 0
    for transformer in transformers:
        num_meters = random.randint(5, 10)
        
        for i in range(num_meters):
            meter_count += 1
            meter = CashPowerMeter.objects.create(
                meter_id=f"METER_{meter_count:03d}",
                transformer=transformer,
                installation_date=datetime.now() - timedelta(days=random.randint(30, 365)),
                is_active=True
            )
            print(f"Created meter: {meter.meter_id} for transformer {transformer.transformer_id}")
    
    print(f"Created {meter_count} sample meters")
    return meter_count

def generate_sample_readings(days=7):
    """Generate sample readings for all meters"""
    print(f"Generating sample readings for the last {days} days...")
    
    meters = CashPowerMeter.objects.filter(is_active=True)
    reading_count = 0
    
    for meter in meters:
        # Base consumption pattern for this meter
        base_consumption = random.uniform(30, 150)  # kWh per day
        
        # Randomly make some meters "thieves"
        is_thief = random.random() < 0.2
        
        for day in range(days):
            for hour in range(24):
                timestamp = datetime.now() - timedelta(days=days-day, hours=hour)
                
                # Generate consumption based on time and theft status
                if is_thief:
                    # Thieves have reduced or zero consumption
                    if random.random() < 0.3:  # 30% chance of zero consumption
                        consumption = 0
                    else:
                        consumption = (base_consumption / 24) * random.uniform(0.2, 0.6)
                else:
                    # Normal consumption pattern
                    hour_factor = 1.0
                    if 6 <= hour <= 9 or 18 <= hour <= 22:  # Peak hours
                        hour_factor = random.uniform(1.2, 1.5)
                    elif 0 <= hour <= 6:  # Off-peak
                        hour_factor = random.uniform(0.3, 0.6)
                    
                    consumption = (base_consumption / 24) * hour_factor * random.uniform(0.8, 1.2)
                
                # Generate electrical parameters
                voltage = random.normalvariate(230, 5)
                current = consumption * 1000 / (voltage * 0.9) if consumption > 0 else 0.1
                power_factor = random.uniform(0.85, 0.95) if consumption > 0 else 0.1
                
                # Expected consumption (what it should be)
                expected_consumption = (base_consumption / 24) * (
                    1.3 if 6 <= hour <= 9 or 18 <= hour <= 22 else 
                    0.5 if 0 <= hour <= 6 else 1.0
                )
                
                reading = PowerMeterReading.objects.create(
                    meter=meter,
                    timestamp=timestamp,
                    energy_consumed_kwh=max(0, consumption),
                    expected_consumption_kwh=expected_consumption,
                    voltage_v=voltage,
                    current_a=current,
                    power_factor=power_factor,
                    reading_source='automated'
                )
                
                reading_count += 1
    
    print(f"Generated {reading_count} sample readings")
    return reading_count

def main():
    """Main function to populate sample data"""
    print("Populating sample meter data...")
    
    # Create meters
    meter_count = create_sample_meters()
    
    # Generate readings
    reading_count = generate_sample_readings(days=7)
    
    print(f"\nSummary:")
    print(f"- Created {meter_count} meters")
    print(f"- Generated {reading_count} readings")
    print(f"- Ready for meter-level theft detection!")

if __name__ == "__main__":
    main()
