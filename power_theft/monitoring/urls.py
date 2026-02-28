from django.urls import path
from . import views
from . import location_views

urlpatterns = [
    # Dashboard and stats
    path('dashboard/stats/', views.dashboard_stats, name='dashboard_stats'),
    
    # Transformers
    path('transformers/', views.transformer_list, name='transformer_list'),
    
    # Predictions
    path('predictions/', views.theft_predictions, name='theft_predictions'),
    path('predict/', views.PredictTheft.as_view(), name='predict_theft'),
    
    # Alerts
    path('alerts/high-risk/', views.high_risk_alerts, name='high_risk_alerts'),
    
    # Meter-level endpoints
    path('meters/', views.meter_list, name='meter_list'),
    path('meters/readings/', views.meter_readings, name='meter_readings'),
    path('meters/detect/', views.MeterTheftDetection.as_view(), name='meter_theft_detection'),
    path('meters/alerts/', views.meter_theft_alerts, name='meter_theft_alerts'),
    path('meters/alerts/<int:alert_id>/resolve/', views.resolve_meter_alert, name='resolve_meter_alert'),
    
    # Location-based endpoints
    path('location/dashboard/', location_views.location_dashboard, name='location_dashboard'),
    path('location/transformers/', location_views.transformers_by_location, name='transformers_by_location'),
    path('location/map-data/', location_views.theft_map_data, name='theft_map_data'),
    path('location/report-incident/', location_views.ReportTheftIncident.as_view(), name='report_theft_incident'),
    path('location/analytics/', location_views.location_analytics, name='location_analytics'),
]
