from django.contrib import admin
from .models import (
    Transformer,
    CashPowerMeter,
    TransformerReading,
    TheftPrediction
)

@admin.register(Transformer)
class TransformerAdmin(admin.ModelAdmin):
    list_display = ('transformer_id', 'location', 'capacity_kva')

@admin.register(TransformerReading)
class TransformerReading(admin.ModelAdmin):
    list_display = ('transformer',
        'timestamp',
        'energy_supplied_kwh',
        'loss_ratio',
        'theft_flag')
    list_filter = ('theft_flag', 'transformer')