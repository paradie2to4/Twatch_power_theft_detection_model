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
from django.db.models import Count, Avg, Sum, Q
from django.db.models.functions import TruncDate, TruncMonth

from .models import (
    Transformer, TheftPrediction, CashPowerMeter, PowerMeterReading, 
    MeterTheftAlert, LocationTheftIncident, TheftHotspot,
    TransformerBehaviorAlert
)
from .load_data import prepare_training_data
from .features import create_features
from .models_ml import random_forest_model
from .meter_features import create_meter_features, detect_meter_anomalies, calculate_theft_probability
from .meter_train import predict_meter_theft
from .transformer_behavior_features import detect_transformer_behavioral_anomalies
from .transformer_behavior_train import predict_transformer_behavior
from sklearn.preprocessing import StandardScaler


@api_view(['GET'])
def location_dashboard(request):
    """Get location-based theft dashboard statistics"""
    try:
        # Overall statistics
        total_transformers = Transformer.objects.count()
        transformers_with_location = Transformer.objects.filter(
            Q(latitude__isnull=False) & Q(longitude__isnull=False)
        ).count()
        
        # Theft incidents by location
        incidents_by_district = LocationTheftIncident.objects.values(
            'district', 'province'
        ).annotate(
            incident_count=Count('id'),
            total_loss=Sum('estimated_loss_usd')
        ).order_by('-incident_count')
        
        # Hotspots
        hotspots = TheftHotspot.objects.all().order_by('-incidents_last_30_days')
        
        # Recent incidents with locations
        recent_incidents = LocationTheftIncident.objects.select_related(
            'transformer'
        ).filter(
            incident_date__gte=datetime.now() - timedelta(days=7)
        ).order_by('-incident_date')[:20]
        
        # Theft alerts with transformer locations
        recent_alerts = TransformerBehaviorAlert.objects.select_related(
            'transformer'
        ).filter(
            timestamp__gte=datetime.now() - timedelta(days=7),
            anomaly_score__gte=50
        ).order_by('-timestamp')[:20]
        
        return Response({
            'summary': {
                'total_transformers': total_transformers,
                'transformers_with_location': transformers_with_location,
                'location_coverage': round((transformers_with_location / total_transformers * 100), 1) if total_transformers > 0 else 0,
                'total_hotspots': hotspots.count(),
                'critical_hotspots': hotspots.filter(risk_level='critical').count(),
                'high_risk_hotspots': hotspots.filter(risk_level='high').count()
            },
            'incidents_by_district': list(incidents_by_district),
            'hotspots': [
                {
                    'area_name': hotspot.area_name,
                    'district': hotspot.district,
                    'province': hotspot.province,
                    'risk_level': hotspot.risk_level,
                    'incidents_last_30_days': hotspot.incidents_last_30_days,
                    'incidents_last_90_days': hotspot.incidents_last_90_days,
                    'total_incidents': hotspot.total_incidents,
                    'total_estimated_loss_usd': hotspot.total_estimated_loss_usd,
                    'center_latitude': float(hotspot.center_latitude) if hotspot.center_latitude else None,
                    'center_longitude': float(hotspot.center_longitude) if hotspot.center_longitude else None,
                    'radius_km': hotspot.radius_km
                }
                for hotspot in hotspots
            ],
            'recent_incidents': [
                {
                    'id': incident.id,
                    'transformer_id': incident.transformer.transformer_id,
                    'location': incident.location_summary,
                    'latitude': float(incident.latitude) if incident.latitude else None,
                    'longitude': float(incident.longitude) if incident.longitude else None,
                    'district': incident.district,
                    'province': incident.province,
                    'incident_date': incident.incident_date.isoformat(),
                    'theft_type': incident.theft_type,
                    'detection_method': incident.detection_method,
                    'confidence_score': incident.confidence_score,
                    'estimated_loss_usd': incident.estimated_loss_usd,
                    'status': incident.status
                }
                for incident in recent_incidents
            ],
            'recent_alerts': [
                {
                    'id': alert.id,
                    'transformer_id': alert.transformer.transformer_id,
                    'location': alert.transformer.full_location,
                    'latitude': float(alert.transformer.latitude) if alert.transformer.latitude else None,
                    'longitude': float(alert.transformer.longitude) if alert.transformer.longitude else None,
                    'district': alert.transformer.district,
                    'province': alert.transformer.province,
                    'timestamp': alert.timestamp.isoformat(),
                    'anomaly_type': alert.anomaly_type,
                    'anomaly_score': alert.anomaly_score,
                    'confidence_level': alert.confidence_level,
                    'behavioral_change_magnitude': alert.behavioral_change_magnitude
                }
                for alert in recent_alerts
            ]
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def transformers_by_location(request):
    """Get transformers filtered by location parameters"""
    try:
        # Get query parameters
        district = request.GET.get('district')
        province = request.GET.get('province')
        lat_min = request.GET.get('lat_min')
        lat_max = request.GET.get('lat_max')
        lng_min = request.GET.get('lng_min')
        lng_max = request.GET.get('lng_max')
        risk_level = request.GET.get('risk_level')
        
        queryset = Transformer.objects.all()
        
        # Apply location filters
        if district:
            queryset = queryset.filter(district__icontains=district)
        if province:
            queryset = queryset.filter(province__icontains=province)
        
        # Apply bounding box filter
        if lat_min and lat_max and lng_min and lng_max:
            queryset = queryset.filter(
                latitude__gte=lat_min,
                latitude__lte=lat_max,
                longitude__gte=lng_min,
                longitude__lte=lng_max
            )
        
        # Filter by transformers with location data
        queryset = queryset.filter(
            latitude__isnull=False,
            longitude__isnull=False
        )
        
        transformers = queryset
        
        data = []
        for transformer in transformers:
            # Get latest behavior alert
            latest_alert = TransformerBehaviorAlert.objects.filter(
                transformer=transformer
            ).order_by('-timestamp').first()
            
            # Get theft incidents count
            incident_count = LocationTheftIncident.objects.filter(
                transformer=transformer
            ).count()
            
            transformer_data = {
                'transformer_id': transformer.transformer_id,
                'location': transformer.full_location,
                'latitude': float(transformer.latitude),
                'longitude': float(transformer.longitude),
                'address': transformer.address,
                'district': transformer.district,
                'province': transformer.province,
                'capacity_kva': transformer.capacity_kva,
                'status': transformer.status,
                'installation_date': transformer.installation_date.isoformat() if transformer.installation_date else None,
                'incident_count': incident_count
            }
            
            if latest_alert:
                transformer_data.update({
                    'latest_alert': {
                        'timestamp': latest_alert.timestamp.isoformat(),
                        'anomaly_type': latest_alert.anomaly_type,
                        'anomaly_score': latest_alert.anomaly_score,
                        'confidence_level': latest_alert.confidence_level
                    }
                })
            
            data.append(transformer_data)
        
        return Response({
            'transformers': data,
            'count': len(data),
            'filters_applied': {
                'district': district,
                'province': province,
                'bounding_box': {
                    'lat_min': lat_min,
                    'lat_max': lat_max,
                    'lng_min': lng_min,
                    'lng_max': lng_max
                }
            }
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def theft_map_data(request):
    """Get data for theft visualization map"""
    try:
        # Get date range
        days = int(request.GET.get('days', 30))
        start_date = datetime.now() - timedelta(days=days)
        
        # Get incidents with locations
        incidents = LocationTheftIncident.objects.filter(
            incident_date__gte=start_date,
            latitude__isnull=False,
            longitude__isnull=False
        ).select_related('transformer')
        
        # Get behavior alerts with locations
        alerts = TransformerBehaviorAlert.objects.filter(
            timestamp__gte=start_date,
            transformer__latitude__isnull=False,
            transformer__longitude__isnull=False,
            anomaly_score__gte=30  # Only significant alerts
        ).select_related('transformer')
        
        # Prepare incident data
        incident_data = []
        for incident in incidents:
            incident_data.append({
                'id': incident.id,
                'type': 'incident',
                'transformer_id': incident.transformer.transformer_id,
                'latitude': float(incident.latitude),
                'longitude': float(incident.longitude),
                'title': f"Theft Incident: {incident.theft_type}",
                'description': f"{incident.theft_type} detected at {incident.location_summary}",
                'date': incident.incident_date.isoformat(),
                'severity': 'high' if incident.estimated_loss_usd and incident.estimated_loss_usd > 1000 else 'medium',
                'estimated_loss_usd': incident.estimated_loss_usd,
                'status': incident.status,
                'popup_content': {
                    'incident_id': incident.id,
                    'theft_type': incident.theft_type,
                    'detection_method': incident.detection_method,
                    'confidence_score': incident.confidence_score,
                    'location': incident.location_summary
                }
            })
        
        # Prepare alert data
        alert_data = []
        for alert in alerts:
            alert_data.append({
                'id': alert.id,
                'type': 'alert',
                'transformer_id': alert.transformer.transformer_id,
                'latitude': float(alert.transformer.latitude),
                'longitude': float(alert.transformer.longitude),
                'title': f"Behavioral Alert: {alert.anomaly_type}",
                'description': f"{alert.anomaly_type} detected with score {alert.anomaly_score:.1f}",
                'date': alert.timestamp.isoformat(),
                'severity': 'critical' if alert.anomaly_score >= 80 else 'high' if alert.anomaly_score >= 60 else 'medium',
                'anomaly_score': alert.anomaly_score,
                'confidence_level': alert.confidence_level,
                'popup_content': {
                    'alert_id': alert.id,
                    'anomaly_type': alert.anomaly_type,
                    'anomaly_score': alert.anomaly_score,
                    'confidence_level': alert.confidence_level,
                    'location': alert.transformer.full_location,
                    'behavioral_change_magnitude': alert.behavioral_change_magnitude
                }
            })
        
        # Combine all data
        all_data = incident_data + alert_data
        
        # Sort by date (most recent first)
        all_data.sort(key=lambda x: x['date'], reverse=True)
        
        return Response({
            'data': all_data,
            'summary': {
                'total_incidents': len(incident_data),
                'total_alerts': len(alert_data),
                'date_range': {
                    'start_date': start_date.isoformat(),
                    'end_date': datetime.now().isoformat(),
                    'days': days
                }
            }
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@extend_schema(
    request={
        'application/json': {
            'type': 'object',
            'properties': {
                'transformer_id': {'type': 'string', 'example': 'TRANS_001'},
                'incident_date': {'type': 'string', 'format': 'date-time', 'example': '2024-01-15T10:30:00'},
                'theft_type': {'type': 'string', 'example': 'energy_theft'},
                'detection_method': {'type': 'string', 'example': 'behavioral_analysis'},
                'confidence_score': {'type': 'number', 'example': 0.85},
                'estimated_loss_kwh': {'type': 'number', 'example': 500.0},
                'estimated_loss_usd': {'type': 'number', 'example': 50.0},
                'investigation_notes': {'type': 'string', 'example': 'Physical inspection confirmed tampered seals'}
            },
            'required': ['transformer_id', 'incident_date', 'theft_type', 'detection_method', 'confidence_score']
        }
    }
)
class ReportTheftIncident(APIView):
    """Report a theft incident at a specific location"""
    
    def post(self, request):
        try:
            data = request.data
            
            # Validate required fields
            required_fields = ['transformer_id', 'incident_date', 'theft_type', 'detection_method', 'confidence_score']
            for field in required_fields:
                if field not in data:
                    return Response({'error': f'Missing required field: {field}'}, status=400)
            
            # Get transformer
            try:
                transformer = Transformer.objects.get(transformer_id=data['transformer_id'])
            except Transformer.DoesNotExist:
                return Response({'error': f'Transformer {data["transformer_id"]} not found'}, status=404)
            
            # Create theft incident
            incident = LocationTheftIncident.objects.create(
                transformer=transformer,
                incident_date=pd.to_datetime(data['incident_date']),
                theft_type=data['theft_type'],
                detection_method=data['detection_method'],
                confidence_score=float(data['confidence_score']),
                estimated_loss_kwh=data.get('estimated_loss_kwh'),
                estimated_loss_usd=data.get('estimated_loss_usd'),
                investigation_notes=data.get('investigation_notes'),
                
                # Copy location from transformer
                latitude=transformer.latitude,
                longitude=transformer.longitude,
                address=transformer.address,
                district=transformer.district,
                province=transformer.province
            )
            
            # Update hotspot if exists
            if transformer.district and transformer.province:
                hotspot, created = TheftHotspot.objects.get_or_create(
                    district=transformer.district,
                    province=transformer.province,
                    area_name=f"{transformer.district} Area",
                    defaults={
                        'center_latitude': transformer.latitude,
                        'center_longitude': transformer.longitude
                    }
                )
                
                # Update hotspot metrics
                hotspot.total_incidents += 1
                if incident.estimated_loss_usd:
                    hotspot.total_estimated_loss_usd += incident.estimated_loss_usd
                    hotspot.average_loss_per_incident = hotspot.total_estimated_loss_usd / hotspot.total_incidents
                hotspot.last_incident_date = incident.incident_date
                hotspot.update_risk_level()
            
            return Response({
                'message': 'Theft incident reported successfully',
                'incident_id': incident.id,
                'location': incident.location_summary,
                'transformer_id': incident.transformer.transformer_id,
                'incident_date': incident.incident_date.isoformat(),
                'theft_type': incident.theft_type,
                'confidence_score': incident.confidence_score
            })
            
        except Exception as e:
            return Response({'error': str(e)}, status=500)


@api_view(['GET'])
def location_analytics(request):
    """Get detailed analytics for theft by location"""
    try:
        # Get date range
        days = int(request.GET.get('days', 90))
        start_date = datetime.now() - timedelta(days=days)
        
        # Incidents by district
        district_stats = LocationTheftIncident.objects.filter(
            incident_date__gte=start_date
        ).values('district', 'province').annotate(
            incident_count=Count('id'),
            total_loss=Sum('estimated_loss_usd'),
            avg_confidence=Avg('confidence_score')
        ).order_by('-incident_count')
        
        # Monthly trend
        monthly_trend = LocationTheftIncident.objects.filter(
            incident_date__gte=start_date
        ).annotate(
            month=TruncMonth('incident_date')
        ).values('month').annotate(
            incident_count=Count('id'),
            total_loss=Sum('estimated_loss_usd')
        ).order_by('month')
        
        # Theft types distribution
        theft_types = LocationTheftIncident.objects.filter(
            incident_date__gte=start_date
        ).values('theft_type').annotate(
            count=Count('id'),
            total_loss=Sum('estimated_loss_usd')
        ).order_by('-count')
        
        # Detection methods
        detection_methods = LocationTheftIncident.objects.filter(
            incident_date__gte=start_date
        ).values('detection_method').annotate(
            count=Count('id'),
            avg_confidence=Avg('confidence_score')
        ).order_by('-count')
        
        return Response({
            'period': {
                'start_date': start_date.isoformat(),
                'end_date': datetime.now().isoformat(),
                'days': days
            },
            'district_stats': list(district_stats),
            'monthly_trend': [
                {
                    'month': item['month'].isoformat(),
                    'incident_count': item['incident_count'],
                    'total_loss': item['total_loss']
                }
                for item in monthly_trend
            ],
            'theft_types': list(theft_types),
            'detection_methods': list(detection_methods)
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)
