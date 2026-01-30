from django.db import models

class Transformer(models.Model):
    transformer_id = models.CharField(max_length= 50, unique=True)
    location = models.CharField(max_length=100, blank=True, null=True) 
    capacity_kva = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.transformer_id

class CashPowerMeter(models.Model):
    meter_id = models.CharField(
        max_length=50,
        unique=True
    )
    transformer = models.ForeignKey(
        Transformer,
        on_delete=models.CASCADE,
        related_name='meters'
    )

    installation_date = models.DateField()
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.meter_id



class TransformerReading(models.Model):
    transformer = models.ForeignKey(
        Transformer,
        on_delete=models.CASCADE,
        related_name='readings'
    )

    timestamp = models.DateTimeField()

    # Core energy values
    energy_supplied_kwh = models.FloatField()
    total_meter_consumption_kwh = models.FloatField()

    # Derived values (from fake data generator)
    loss_kwh = models.FloatField()
    loss_ratio = models.FloatField()

    # Ground truth (for training)
    theft_flag = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['transformer']),
            models.Index(fields=['theft_flag']),
        ]

    def __str__(self):
        return f"{self.transformer} @ {self.timestamp}"

class TheftPrediction(models.Model):
    transformer = models.ForeignKey(
        Transformer,
        on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField()

    loss_ratio = models.FloatField()
    theft_probability = models.FloatField()

    theft_detected = models.BooleanField()

    model_version = models.CharField(max_length=20)

    created_at = models.DateTimeField(auto_now_add=True)
