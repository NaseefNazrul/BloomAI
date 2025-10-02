import ee
import time
from datetime import datetime, timedelta

def initialize_earth_engine():
    """Initialize Earth Engine with error handling"""
    try:
        ee.Initialize(project='stable-ring-473811-a3')
        print("✅ Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Earth Engine initialization failed: {e}")
        return False

def get_essential_vegetation_data(lat, lon, target_date, buffer_meters=200, max_days_back=30):
    """
    Get essential vegetation data optimized for ML bloom prediction
    Returns only the features needed for your model
    """
    point = ee.Geometry.Point([lon, lat])
    area = point.buffer(buffer_meters)
    
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Extract date features for ML
    date_features = extract_date_features(target_dt)
    
    # Try Landsat 9 first with date fallback
    satellite_data = get_satellite_data_with_fallback(
        lat, lon, target_dt, 'Landsat-9', buffer_meters, area, max_days_back
    )
    
    if not satellite_data:
        # Fall back to Landsat 8
        satellite_data = get_satellite_data_with_fallback(
            lat, lon, target_dt, 'Landsat-8', buffer_meters, area, max_days_back
        )
    
    if not satellite_data:
        return None
    
    # Combine all features for ML
    ml_features = {
        **satellite_data,
        **date_features,
        'longitude': lon,
        'latitude': lat
    }
    
    return ml_features

def get_satellite_data_with_fallback(lat, lon, target_dt, satellite, buffer_meters, area, max_days_back):
    """Get satellite data with date fallback for a specific satellite"""
    
    for days_back in range(0, max_days_back + 1):
        current_date = (target_dt - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        data = get_single_date_satellite_data(lat, lon, current_date, satellite, buffer_meters, area)
        
        if data and data['ndvi'] is not None:
            print(f"✅ {satellite} data found for {current_date} ({days_back} days back)")
            
            # Add fallback info if we used a different date
            if days_back > 0:
                data['original_request_date'] = target_dt.strftime('%Y-%m-%d')
                data['actual_data_date'] = current_date
                data['days_offset'] = days_back
            
            return data
    
    return None

def get_single_date_satellite_data(lat, lon, date, satellite, buffer_meters, area):
    """Get vegetation data for a single date and satellite"""
    
    collection_id = 'LANDSAT/LC09/C02/T1_L2' if satellite == 'Landsat-9' else 'LANDSAT/LC08/C02/T1_L2'
    
    try:
        # Efficient filtering - single day range
        filtered = (ee.ImageCollection(collection_id)
                   .filterBounds(area)
                   .filterDate(date, f'{date}T23:59:59')
                   .sort('CLOUD_COVER')
                   .limit(1))
        
        if filtered.size().getInfo() == 0:
            return None
        
        image = filtered.first()
        image_info = image.getInfo()['properties']
        cloud_cover = image_info.get('CLOUD_COVER', 100)
        
        # Skip if heavy cloud cover
        if cloud_cover > 80:
            return None
        
        # Calculate essential indices only
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
        composite = ndvi.addBands(ndwi)
        
        # Single efficient API call for both indices
        area_stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area,
            scale=30,
            maxPixels=1e6,
            bestEffort=True
        ).getInfo()
        
        ndvi_val = area_stats.get('NDVI')
        ndwi_val = area_stats.get('NDWI')
        
        if ndvi_val is None:
            return None
        
        return {
            'ndvi': float(ndvi_val),
            'ndwi': float(ndwi_val),
            'cloud_cover': float(cloud_cover),
            'satellite': satellite,
            'date': date,
            'buffer_size': buffer_meters
        }
        
    except Exception as e:
        print(f"❌ Error getting {satellite} data: {e}")
        return None

def extract_date_features(date_obj):
    """Extract seasonal features for ML model"""
    month = date_obj.month
    day_of_year = date_obj.timetuple().tm_yday
    day_of_month = date_obj.day
    
    # Seasonal indicators (adjust ranges based on your region)
    is_spring = 1 if 3 <= month <= 5 else 0      # March-May
    is_summer = 1 if 6 <= month <= 8 else 0      # June-August  
    is_fall = 1 if 9 <= month <= 11 else 0       # September-November
    is_winter = 1 if month in [12, 1, 2] else 0  # December-February
    
    return {
        'month': month,
        'day_of_year': day_of_year,
        'day_of_month': day_of_month,
        'is_spring': is_spring,
        'is_summer': is_summer,
        'is_fall': is_fall,
        'is_winter': is_winter
    }

def calculate_bloom_probability_ml(ml_features):
    """
    Calculate bloom probability using ML features
    This is where your actual ML model will integrate
    """
    # Placeholder for your ML model - replace with actual model prediction
    # For now, using a simple rule-based approach
    
    ndvi = ml_features['ndvi']
    ndwi = ml_features['ndwi']
    is_spring = ml_features['is_spring']
    
    # Simple scoring (replace with your trained ML model)
    score = 0.0
    
    # NDVI contribution (vegetation health)
    if ndvi > 0.6:
        score += 0.6
    elif ndvi > 0.4:
        score += 0.4
    elif ndvi > 0.2:
        score += 0.2
    
    # NDWI contribution (water content)
    if ndwi > 0.2:
        score += 0.3
    elif ndwi > 0.0:
        score += 0.15
    
    # Seasonal multiplier
    if is_spring:
        score *= 1.3  # Spring boost
    elif ml_features['is_winter']:
        score *= 0.4  # Winter reduction
    
    # Ensure probability is between 0-100%
    probability = min(95, max(0, score * 100))
    
    return probability

# Main API function
def get_bloom_analysis(lat, lon, date):
    """
    Main function for your bloom probability API
    Returns optimized JSON for frontend
    """
    start_time = time.time()
    
    print(f"🌼 Starting bloom analysis for {lat}, {lon} on {date}")
    
    # Get ML-ready features
    ml_features = get_essential_vegetation_data(lat, lon, date)
    
    if not ml_features:
        return {
            'error': 'No satellite data available for this location and time period',
            'bloom_probability': 0.0,
            'confidence': 'low',
            'message': 'Unable to analyze due to missing satellite data'
        }
    
    # Calculate probability using ML features
    probability = calculate_bloom_probability_ml(ml_features)
    
    # Prepare clean JSON response for frontend
    response = {
        'bloom_probability': round(probability, 2),
        'analysis_date': ml_features.get('actual_data_date', ml_features['date']),
        'requested_date': date,
        'data_quality': {
            'satellite': ml_features['satellite'],
            'cloud_cover': ml_features['cloud_cover'],
            'days_old': ml_features.get('days_offset', 0)
        },
        'vegetation_health': {
            'ndvi': ml_features['ndvi'],
            'ndwi': ml_features['ndwi'],
            'assessment': 'high' if ml_features['ndvi'] > 0.6 else 'medium' if ml_features['ndvi'] > 0.4 else 'low'
        },
        'seasonal_context': {
            'season': 'spring' if ml_features['is_spring'] else 'summer' if ml_features['is_summer'] else 'fall' if ml_features['is_fall'] else 'winter',
            'month': ml_features['month']
        },
        'processing_time': round(time.time() - start_time, 2)
    }
    
    # Add appropriate message based on probability
    if probability > 70:
        response['message'] = 'High likelihood of wildflower blooms!'
    elif probability > 40:
        response['message'] = 'Moderate chance of wildflower blooms'
    else:
        response['message'] = 'Low probability of blooms at this time'
    
    print(f"✅ Analysis complete - {response['message']}")
    return response

# Example usage and testing
if __name__ == "__main__":
    if initialize_earth_engine():
        # Test the optimized pipeline
        test_lat = 33.276734080468145
        test_lon = -110.30621859902435
        test_date = '2023-04-28'  # Spring date for testing
        
        result = get_bloom_analysis(test_lat, test_lon, test_date)
        print("\n🎯 BLOOM ANALYSIS RESULT:")
        print(f"Bloom Probability: {result['bloom_probability']}%")
        print(f"Message: {result['message']}")
        print(f"Season: {result['seasonal_context']['season']}")
        print(f"NDVI: {result['vegetation_health']['ndvi']}")
        print(f"Processing Time: {result['processing_time']}s")