from django.urls import path
from . import views

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
]
