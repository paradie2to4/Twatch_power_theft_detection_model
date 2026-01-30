import os
import pandas as pd
from django.conf import settings

def get_data_path():
    """Get data path - works in both development and production"""
    if 'RENDER' in os.environ:
        # Production: Use Render disk or external storage
        data_path = '/opt/render/project/data'
    else:
        # Development: Use local data folder
        data_path = os.path.join(settings.BASE_DIR, 'data')
    
    return data_path

def ensure_data_exists():
    """Ensure data files exist, download if necessary"""
    data_path = get_data_path()
    
    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)
    
    # Check if files exist, if not, you could:
    # 1. Download from external source
    # 2. Use sample data
    # 3. Return empty DataFrames
    
    required_files = [
        'datalarge_meters.csv',
        'datalarge_meter_readings.csv', 
        'datalarge_transformer_readings.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(data_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing data files: {missing_files}")
        print("Please upload data files to the server or use sample data")
        return False
    
    return True
