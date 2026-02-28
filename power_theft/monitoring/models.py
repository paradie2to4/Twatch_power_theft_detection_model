from django.db import models

class Transformer(models.Model):
    transformer_id = models.CharField(max_length= 50, unique=True)
    
    # Location information
    location = models.CharField(max_length=100, blank=True, null=True)
    latitude = models.DecimalField(max_digits=10, decimal_places=8, null=True, blank=True)
    longitude = models.DecimalField(max_digits=11, decimal_places=8, null=True, blank=True)
    address = models.TextField(blank=True, null=True)
    district = models.CharField(max_length=100, blank=True, null=True)
    province = models.CharField(max_length=100, blank=True, null=True)
    postal_code = models.CharField(max_length=20, blank=True, null=True)
    
    # Technical specifications
    capacity_kva = models.FloatField()
    installation_date = models.DateField(null=True, blank=True)
    transformer_type = models.CharField(max_length=50, blank=True, null=True)  # Pole-mounted, pad-mounted, etc.
    manufacturer = models.CharField(max_length=100, blank=True, null=True)
    serial_number = models.CharField(max_length=100, blank=True, null=True)
    
    # Status information
    status = models.CharField(max_length=20, choices=[
        ('active', 'Active'),
        ('maintenance', 'Under Maintenance'),
        ('decommissioned', 'Decommissioned'),
        ('fault', 'Fault')
    ], default='active')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.transformer_id} - {self.location or 'No Location'}"
    
    @property
    def full_location(self):
        """Return formatted full location"""
        parts = []
        if self.address:
            parts.append(self.address)
        if self.district:
            parts.append(self.district)
        if self.province:
            parts.append(self.province)
        return ", ".join(parts) if parts else "Location not specified"
    
    @property
    def coordinates(self):
        """Return coordinates as tuple"""
        if self.latitude and self.longitude:
            return (float(self.latitude), float(self.longitude))
        return None

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

    # Electrical parameters for behavioral analysis
    output_current_a = models.FloatField(null=True, blank=True)  # Output current in Amperes
    output_voltage_v = models.FloatField(null=True, blank=True)  # Output voltage in Volts
    power_factor = models.FloatField(null=True, blank=True)  # Power factor
    frequency_hz = models.FloatField(null=True, blank=True)  # Grid frequency

    # Derived values (from fake data generator)
    loss_kwh = models.FloatField()
    loss_ratio = models.FloatField()

    # Behavioral change detection metrics
    current_variance = models.FloatField(null=True, blank=True)  # Rolling variance of current
    voltage_variance = models.FloatField(null=True, blank=True)  # Rolling variance of voltage
    current_rate_of_change = models.FloatField(null=True, blank=True)  # ΔI/Δt
    voltage_rate_of_change = models.FloatField(null=True, blank=True)  # ΔV/Δt

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


class LocationTheftIncident(models.Model):
    """Track theft incidents by location for geographic analysis"""
    transformer = models.ForeignKey(
        Transformer,
        on_delete=models.CASCADE,
        related_name='theft_incidents'
    )
    
    # Location information (denormalized for quick queries)
    latitude = models.DecimalField(max_digits=10, decimal_places=8, null=True, blank=True)
    longitude = models.DecimalField(max_digits=11, decimal_places=8, null=True, blank=True)
    address = models.TextField(blank=True, null=True)
    district = models.CharField(max_length=100, blank=True, null=True)
    province = models.CharField(max_length=100, blank=True, null=True)
    
    # Incident details
    incident_date = models.DateTimeField()
    theft_type = models.CharField(max_length=50, choices=[
        ('energy_theft', 'Energy Theft'),
        ('meter_tampering', 'Meter Tampering'),
        ('bypass', 'Direct Bypass'),
        ('illegal_connection', 'Illegal Connection')
    ])
    
    # Detection information
    detection_method = models.CharField(max_length=50, choices=[
        ('behavioral_analysis', 'Behavioral Analysis'),
        ('manual_inspection', 'Manual Inspection'),
        ('customer_complaint', 'Customer Complaint'),
        ('routine_audit', 'Routine Audit')
    ])
    
    confidence_score = models.FloatField()  # 0.0 to 1.0
    estimated_loss_kwh = models.FloatField(null=True, blank=True)
    estimated_loss_usd = models.FloatField(null=True, blank=True)
    
    # Status and resolution
    status = models.CharField(max_length=20, choices=[
        ('reported', 'Reported'),
        ('investigating', 'Investigating'),
        ('confirmed', 'Confirmed'),
        ('resolved', 'Resolved'),
        ('false_positive', 'False Positive')
    ], default='reported')
    
    investigation_notes = models.TextField(blank=True, null=True)
    resolved_by = models.CharField(max_length=100, blank=True, null=True)
    resolution_date = models.DateTimeField(null=True, blank=True)
    recovery_amount_usd = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['transformer', 'incident_date']),
            models.Index(fields=['district', 'incident_date']),
            models.Index(fields=['province', 'incident_date']),
            models.Index(fields=['theft_type']),
            models.Index(fields=['status']),
            models.Index(fields=['incident_date']),
        ]
    
    def __str__(self):
        return f"{self.transformer.transformer_id} - {self.theft_type} at {self.district or 'Unknown Location'}"
    
    @property
    def location_summary(self):
        """Return concise location summary"""
        if self.district and self.province:
            return f"{self.district}, {self.province}"
        elif self.district:
            return self.district
        elif self.address:
            return self.address[:50] + "..." if len(self.address) > 50 else self.address
        return "Location not specified"


class TheftHotspot(models.Model):
    """Identify and track theft hotspots by geographic area"""
    
    # Geographic area definition
    district = models.CharField(max_length=100)
    province = models.CharField(max_length=100)
    area_name = models.CharField(max_length=100)  # e.g., "Kigali Central"
    
    # Geographic boundaries (optional, for mapping)
    center_latitude = models.DecimalField(max_digits=10, decimal_places=8, null=True, blank=True)
    center_longitude = models.DecimalField(max_digits=11, decimal_places=8, null=True, blank=True)
    radius_km = models.FloatField(null=True, blank=True)  # Radius of area in kilometers
    
    # Hotspot metrics
    total_incidents = models.IntegerField(default=0)
    incidents_last_30_days = models.IntegerField(default=0)
    incidents_last_90_days = models.IntegerField(default=0)
    
    # Risk assessment
    risk_level = models.CharField(max_length=20, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical')
    ], default='low')
    
    # Loss estimates
    total_estimated_loss_usd = models.FloatField(default=0)
    average_loss_per_incident = models.FloatField(default=0)
    
    # Timestamps
    last_incident_date = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['district', 'province']),
            models.Index(fields=['risk_level']),
            models.Index(fields=['incidents_last_30_days']),
        ]
        unique_together = ['district', 'province', 'area_name']
    
    def __str__(self):
        return f"{self.area_name} - {self.risk_level} risk ({self.incidents_last_30_days} incidents)"
    
    def update_risk_level(self):
        """Update risk level based on recent incidents"""
        if self.incidents_last_30_days >= 10:
            self.risk_level = 'critical'
        elif self.incidents_last_30_days >= 5:
            self.risk_level = 'high'
        elif self.incidents_last_30_days >= 2:
            self.risk_level = 'medium'
        else:
            self.risk_level = 'low'
        self.save()


class TransformerBehaviorAlert(models.Model):
    """Model for Transformer Behavior Baseline Model outputs"""
    transformer = models.ForeignKey(
        Transformer,
        on_delete=models.CASCADE,
        related_name='behavior_alerts'
    )
    timestamp = models.DateTimeField()
    
    # Anomaly detection outputs
    anomaly_score = models.FloatField()  # 0-100 scale
    anomaly_type = models.CharField(max_length=50, choices=[
        ('load_spike', 'Load Spike'),
        ('off_peak_load', 'Off-Peak Load'),
        ('instability', 'Instability'),
        ('voltage_anomaly', 'Voltage Anomaly'),
        ('current_anomaly', 'Current Anomaly'),
        ('normal', 'Normal')
    ])
    
    confidence_level = models.FloatField()  # 0.0 to 1.0
    
    # Behavioral metrics
    current_deviation = models.FloatField(null=True, blank=True)  # Deviation from normal current
    voltage_deviation = models.FloatField(null=True, blank=True)  # Deviation from normal voltage
    behavioral_change_magnitude = models.FloatField()  # Magnitude of behavioral change
    
    # Contextual information
    is_resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['transformer', 'timestamp']),
            models.Index(fields=['anomaly_type']),
            models.Index(fields=['anomaly_score']),
            models.Index(fields=['is_resolved']),
        ]
    
    def __str__(self):
        return f"{self.transformer.transformer_id} - {self.anomaly_type} ({self.anomaly_score:.1f})"


class PowerMeterReading(models.Model):
    meter = models.ForeignKey(
        CashPowerMeter,
        on_delete=models.CASCADE,
        related_name='readings'
    )
    timestamp = models.DateTimeField()
    
    # Meter consumption data
    energy_consumed_kwh = models.FloatField()
    expected_consumption_kwh = models.FloatField(null=True, blank=True)
    
    # Voltage and current readings for anomaly detection
    voltage_v = models.FloatField(null=True, blank=True)
    current_a = models.FloatField(null=True, blank=True)
    power_factor = models.FloatField(null=True, blank=True)
    
    # Derived metrics
    consumption_deviation = models.FloatField(null=True, blank=True)
    deviation_percentage = models.FloatField(null=True, blank=True)
    
    # Theft detection flags
    anomaly_detected = models.BooleanField(default=False)
    theft_probability = models.FloatField(null=True, blank=True)
    theft_detected = models.BooleanField(default=False)
    
    # Metadata
    reading_source = models.CharField(max_length=50, default='manual')  # manual, automated, scada
    data_quality_flag = models.BooleanField(default=True)  # True for good quality data
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['meter', 'timestamp']),
            models.Index(fields=['timestamp']),
            models.Index(fields=['theft_detected']),
            models.Index(fields=['anomaly_detected']),
        ]
        unique_together = ['meter', 'timestamp']
    
    def __str__(self):
        return f"{self.meter.meter_id} @ {self.timestamp}"
    
    def save(self, *args, **kwargs):
        # Calculate deviation metrics if expected consumption is available
        if self.expected_consumption_kwh is not None:
            self.consumption_deviation = self.expected_consumption_kwh - self.energy_consumed_kwh
            if self.expected_consumption_kwh > 0:
                self.deviation_percentage = (self.consumption_deviation / self.expected_consumption_kwh) * 100
        
        super().save(*args, **kwargs)


class MeterTheftAlert(models.Model):
    meter = models.ForeignKey(
        CashPowerMeter,
        on_delete=models.CASCADE,
        related_name='theft_alerts'
    )
    reading = models.ForeignKey(
        PowerMeterReading,
        on_delete=models.CASCADE,
        related_name='alerts'
    )
    
    alert_type = models.CharField(max_length=50)  # consumption_anomaly, zero_consumption, tampering
    severity = models.CharField(max_length=20, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'), 
        ('high', 'High'),
        ('critical', 'Critical')
    ])
    
    confidence_score = models.FloatField()  # 0.0 to 1.0
    description = models.TextField()
    
    is_resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolution_notes = models.TextField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['meter', 'created_at']),
            models.Index(fields=['severity']),
            models.Index(fields=['is_resolved']),
        ]
    
    def __str__(self):
        return f"{self.alert_type} - {self.meter.meter_id} ({self.severity})"
