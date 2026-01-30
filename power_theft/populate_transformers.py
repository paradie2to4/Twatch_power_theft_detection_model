#!/usr/bin/env python
import os
import django
from monitoring.load_data import load_data
from monitoring.models import Transformer

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "power_theft.settings")
django.setup()

def populate_transformers():
    meters, _, _ = load_data()
    
    created_count = 0
    for _, row in meters.iterrows():
        transformer, created = Transformer.objects.get_or_create(
            transformer_id=str(row['transformer_id']),
            defaults={'capacity_kva': 100.0}
        )
        if created:
            created_count += 1
    
    print(f'Created {created_count} transformers')
    print(f'Total transformers in DB: {Transformer.objects.count()}')

if __name__ == "__main__":
    populate_transformers()
