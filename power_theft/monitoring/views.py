from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema
from drf_spectacular.types import OpenApiTypes
from datetime import datetime, timedelta
import pandas as pd

from .models import Transformer, TheftPrediction
from .load_data import prepare_training_data
from .features import create_features
from .models_ml import random_forest_model
from sklearn.preprocessing import StandardScaler


@api_view(['GET'])
def dashboard_stats(request):
    """Get dashboard statistics"""
    try:
        total_transformers = Transformer.objects.count()
        total_predictions = TheftPrediction.objects.count()
        recent_thefts = TheftPrediction.objects.filter(
            theft_detected=True,
            timestamp__gte=datetime.now() - timedelta(days=7)
        ).count()
        
        # Get high theft probability transformers
        high_risk = TheftPrediction.objects.filter(
            theft_probability__gte=0.8,
            timestamp__gte=datetime.now() - timedelta(days=1)
        ).count()
        
        return Response({
            'total_transformers': total_transformers,
            'total_predictions': total_predictions,
            'recent_thefts': recent_thefts,
            'high_risk_transformers': high_risk,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def theft_predictions(request):
    """Get theft predictions with filtering"""
    try:
        # Get query parameters
        transformer_id = request.GET.get('transformer_id')
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        theft_detected = request.GET.get('theft_detected')
        limit = int(request.GET.get('limit', 100))
        
        queryset = TheftPrediction.objects.select_related('transformer')
        
        # Apply filters
        if transformer_id:
            queryset = queryset.filter(transformer__transformer_id=transformer_id)
        if start_date:
            queryset = queryset.filter(timestamp__gte=start_date)
        if end_date:
            queryset = queryset.filter(timestamp__lte=end_date)
        if theft_detected is not None:
            queryset = queryset.filter(theft_detected=theft_detected.lower() == 'true')
        
        # Get recent records first
        predictions = queryset.order_by('-timestamp')[:limit]
        
        # Format response
        data = []
        for pred in predictions:
            data.append({
                'id': pred.id,
                'transformer_id': pred.transformer.transformer_id,
                'timestamp': pred.timestamp.isoformat(),
                'loss_ratio': pred.loss_ratio,
                'theft_probability': pred.theft_probability,
                'theft_detected': pred.theft_detected,
                'model_version': pred.model_version,
                'created_at': pred.created_at.isoformat()
            })
        
        return Response({
            'predictions': data,
            'count': len(data),
            'filters_applied': {
                'transformer_id': transformer_id,
                'start_date': start_date,
                'end_date': end_date,
                'theft_detected': theft_detected,
                'limit': limit
            }
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def transformer_list(request):
    """Get list of all transformers"""
    try:
        transformers = Transformer.objects.all()
        data = []
        for t in transformers:
            # Get latest prediction for each transformer
            latest_pred = TheftPrediction.objects.filter(
                transformer=t
            ).order_by('-timestamp').first()
            
            data.append({
                'transformer_id': t.transformer_id,
                'capacity_kva': t.capacity_kva,
                'location': t.location,
                'created_at': t.created_at.isoformat(),
                'latest_prediction': {
                    'timestamp': latest_pred.timestamp.isoformat(),
                    'theft_probability': latest_pred.theft_probability,
                    'theft_detected': latest_pred.theft_detected,
                    'loss_ratio': latest_pred.loss_ratio
                } if latest_pred else None
            })
        
        return Response({
            'transformers': data,
            'count': len(data)
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def high_risk_alerts(request):
    """Get high risk theft alerts"""
    try:
        # Get predictions with high theft probability
        high_risk = TheftPrediction.objects.filter(
            theft_probability__gte=0.7,
            timestamp__gte=datetime.now() - timedelta(hours=24)
        ).select_related('transformer').order_by('-theft_probability')
        
        data = []
        for pred in high_risk:
            data.append({
                'transformer_id': pred.transformer.transformer_id,
                'timestamp': pred.timestamp.isoformat(),
                'theft_probability': pred.theft_probability,
                'loss_ratio': pred.loss_ratio,
                'risk_level': 'Critical' if pred.theft_probability >= 0.9 else 'High'
            })
        
        return Response({
            'alerts': data,
            'count': len(data),
            'threshold': 0.7,
            'timeframe': '24 hours'
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@extend_schema(
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'transformer_id': {'type': 'string', 'example': '1'},
                'energy_supplied_kwh': {'type': 'number', 'example': 100.5},
                'energy_consumed_kwh': {'type': 'number', 'example': 85.2},
                'timestamp': {'type': 'string', 'format': 'date-time', 'example': '2024-01-15T10:30:00'}
            },
            'required': ['transformer_id', 'energy_supplied_kwh', 'energy_consumed_kwh', 'timestamp']
        }
    },
    responses={
        200: {
            'type': 'object',
            'properties': {
                'transformer_id': {'type': 'string'},
                'timestamp': {'type': 'string'},
                'theft_probability': {'type': 'number'},
                'theft_detected': {'type': 'boolean'},
                'loss_ratio': {'type': 'number'},
                'excess_loss_ratio': {'type': 'number'},
                'technical_losses_baseline': {'type': 'number'},
                'risk_assessment': {'type': 'string'},
                'features_used': {'type': 'array', 'items': {'type': 'string'}},
                'energy_supplied_kwh': {'type': 'number'},
                'energy_consumed_kwh': {'type': 'number'}
            }
        },
        400: {'type': 'object', 'properties': {'error': {'type': 'string'}}},
        500: {'type': 'object', 'properties': {'error': {'type': 'string'}}}
    },
    description='Detect power theft for a transformer based on energy consumption data',
    summary='Predict power theft'
)
class PredictTheft(APIView):
    """Predict theft for new data"""
    
    def post(self, request):
        try:
            import json
            import pickle
            import os
            from django.conf import settings
            
            data = request.data
            
            # Validate required fields
            required_fields = ['transformer_id', 'energy_supplied_kwh', 'energy_consumed_kwh', 'timestamp']
            for field in required_fields:
                if field not in data:
                    return Response({'error': f'Missing required field: {field}'}, status=400)
            
            # Try to load trained model
            model_path = os.path.join(settings.BASE_DIR, 'model.pkl')
            scaler_path = os.path.join(settings.BASE_DIR, 'scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                # Use trained model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                # Create DataFrame for prediction
                df_data = {
                    'transformer_id': [data['transformer_id']],
                    'timestamp': [pd.to_datetime(data['timestamp'])],
                    'energy_supplied_kwh': [float(data['energy_supplied_kwh'])],
                    'energy_consumed_kwh': [float(data['energy_consumed_kwh'])],
                    'loss_ratio': [(float(data['energy_supplied_kwh']) - float(data['energy_consumed_kwh'])) / float(data['energy_supplied_kwh'])]
                }
                
                df = pd.DataFrame(df_data)
                df = create_features(df)
                
                # Use model for prediction
                FEATURES = [
                    "energy_supplied_kwh",
                    "energy_consumed_kwh", 
                    "loss_ratio",
                    "hour",
                    "weekday",
                    "is_peak_hour",
                    "loss_ratio_24h_avg",
                    "loss_ratio_7d_avg",
                    "supply_24h_avg"
                ]
                
                X = df[FEATURES]
                X_scaled = scaler.transform(X)
                theft_probability = model.predict_proba(X_scaled)[0][1]
                theft_detected = model.predict(X_scaled)[0]
                
                return Response({
                    'transformer_id': data['transformer_id'],
                    'timestamp': data['timestamp'],
                    'theft_probability': round(theft_probability, 3),
                    'theft_detected': bool(theft_detected),
                    'loss_ratio': df['loss_ratio'].iloc[0],
                    'model_used': 'trained_random_forest',
                    'features_used': FEATURES
                })
            else:
                # Fallback to rule-based logic
                return self._rule_based_prediction(data)
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)
    
    def _rule_based_prediction(self, data):
        """Fallback rule-based prediction"""
        # Calculate loss ratio directly
        energy_supplied = float(data['energy_supplied_kwh'])
        energy_consumed = float(data['energy_consumed_kwh'])
        loss_ratio = (energy_supplied - energy_consumed) / energy_supplied if energy_supplied > 0 else 0
        
        # Account for legitimate technical losses (5-10% is normal)
        technical_losses = 0.08  # 8% baseline technical losses
        excess_loss = max(0, loss_ratio - technical_losses)
        
        # Simple time features
        timestamp = pd.to_datetime(data['timestamp'])
        hour = timestamp.hour
        weekday = timestamp.weekday
        is_peak_hour = 1 if 18 <= hour <= 22 else 0
        
        # More realistic theft detection logic
        if loss_ratio <= 0.10:  # <= 10% loss = Normal
            theft_probability = 0.0
            theft_detected = False
        elif loss_ratio <= 0.15:  # 10-15% loss = Monitor
            theft_probability = excess_loss * 3  # Scale excess loss
            theft_detected = False
        elif loss_ratio <= 0.25:  # 15-25% loss = Suspicious
            theft_probability = 0.3 + (excess_loss * 4)
            theft_detected = theft_probability > 0.5
        else:  # > 25% loss = High probability of theft
            theft_probability = min(0.7 + (excess_loss * 2), 1.0)
            theft_detected = True
        
        # Additional risk factors
        if is_peak_hour and loss_ratio > 0.12:
            theft_probability += 0.1  # Higher losses during peak hours are more suspicious
        
        theft_probability = min(theft_probability, 1.0)
        theft_detected = theft_probability > 0.6  # Higher threshold for detection
        
        return Response({
            'transformer_id': data['transformer_id'],
            'timestamp': data['timestamp'],
            'theft_probability': round(theft_probability, 3),
            'theft_detected': theft_detected,
            'loss_ratio': round(loss_ratio, 4),
            'excess_loss_ratio': round(excess_loss, 4),
            'technical_losses_baseline': technical_losses,
            'risk_assessment': self._get_risk_level(loss_ratio),
            'model_used': 'rule_based_fallback',
            'features_used': ['loss_ratio', 'excess_loss', 'hour', 'weekday', 'is_peak_hour'],
            'energy_supplied_kwh': energy_supplied,
            'energy_consumed_kwh': energy_consumed
        })
    
    def _get_risk_level(self, loss_ratio):
        """Helper method to categorize risk level"""
        if loss_ratio <= 0.10:
            return "Normal"
        elif loss_ratio <= 0.15:
            return "Monitor"
        elif loss_ratio <= 0.25:
            return "Suspicious"
        else:
            return "High Risk"
