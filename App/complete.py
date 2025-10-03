import ee
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import os

# Request data model
class BloomPredictionRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude between -90 and 90")
    lon: float = Field(..., ge=-180, le=180, description="Longitude between -180 and 180")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    
    class Config:
        schema_extra = {
            "example": {
                "lat": 33.2767,
                "lon": -110.3062,
                "date": "2023-04-15"
            }
        }

# Response data model
class BloomPredictionResponse(BaseModel):
    success: bool
    bloom_probability: float
    prediction: str
    confidence: str
    message: str
    analysis_date: str
    requested_date: str
    data_quality: dict
    vegetation_indices: dict
    location: dict
    processing_time: float
    recommendation: str = None

# Global variables for ML model and Earth Engine
ML_MODEL = None
SCALER = None
FEATURE_COLUMNS = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load ML model and initialize Earth Engine
    global ML_MODEL, SCALER, FEATURE_COLUMNS
    
    try:
        ML_MODEL = joblib.load('C:/Users/User/Desktop/nasashit/BloomAI/App/mil_bloom_model.joblib')
        SCALER = joblib.load('C:/Users/User/Desktop/nasashit/BloomAI/App/mil_features.joblib')
        FEATURE_COLUMNS = joblib.load('C:/Users/User/Desktop/nasashit/BloomAI/App/mil_scaler.joblib')
        print("✅ ML Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load ML model: {e}")
        ML_MODEL = None
    
    # Initialize Earth Engine
    if not initialize_earth_engine():
        raise RuntimeError("Failed to initialize Earth Engine")
    
    yield  # This is where the app runs
    
    # Shutdown: Cleanup if needed
    print("🔄 Shutting down Bloom Prediction API")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Bloom Prediction API",
    description="Predict wildflower bloom probability using satellite data and machine learning",
    version="1.0.0",
    lifespan=lifespan
)

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
    """
    point = ee.Geometry.Point([lon, lat])
    area = point.buffer(buffer_meters)
    
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    
    # Try Landsat 9 first with date fallback
    print(f"🛰️ Searching Landsat 9 data...")
    satellite_data = get_satellite_data_with_fallback(
        lat, lon, target_dt, 'Landsat-9', buffer_meters, area, max_days_back
    )
    
    if not satellite_data:
        # Fall back to Landsat 8
        print(f"🛰️ Landsat 9 not found, trying Landsat 8...")
        satellite_data = get_satellite_data_with_fallback(
            lat, lon, target_dt, 'Landsat-8', buffer_meters, area, max_days_back
        )
    
    if not satellite_data:
        return None
    
    # Add location to features
    satellite_data['longitude'] = lon
    satellite_data['latitude'] = lat
    
    return satellite_data

def get_satellite_data_with_fallback(lat, lon, target_dt, satellite, buffer_meters, area, max_days_back):
    """Get satellite data with date fallback for a specific satellite"""
    
    for days_back in range(0, max_days_back + 1):
        current_date = (target_dt - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        data = get_single_date_satellite_data(lat, lon, current_date, satellite, buffer_meters, area)
        
        if data and data['ndvi'] is not None:
            print(f"✅ {satellite} data found for {current_date} ({days_back} days back)")
            
            # Add metadata about date fallback
            data['original_request_date'] = target_dt.strftime('%Y-%m-%d')
            data['actual_data_date'] = current_date
            data['days_offset'] = days_back
            
            return data
    
    print(f"❌ No {satellite} data found within {max_days_back} days")
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
        
        # Skip if heavy cloud cover (>80%)
        if cloud_cover > 80:
            print(f"   Skipping {date}: High cloud cover ({cloud_cover}%)")
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
        print(f"   Error getting {satellite} data for {date}: {e}")
        return None

def predict_bloom_with_ml(features_dict):
    """
    FIXED ML prediction function with better error handling
    """
    
    # if ML_MODEL is None:
    #     print("⚠️ ML model not loaded, using fallback heuristic")
    #     return predict_bloom_fallback(features_dict)
    
    # try:
    #     # DEBUG: Check what we're feeding the model
    #     print(f"🔍 ML Input Debug:")
    #     print(f"   NDVI: {features_dict['ndvi']} (type: {type(features_dict['ndvi'])})")
    #     print(f"   NDWI: {features_dict['ndwi']} (type: {type(features_dict['ndwi'])})")
    #     print(f"   Cloud Cover: {features_dict['cloud_cover']} (type: {type(features_dict['cloud_cover'])})")
    #     print(f"   Latitude: {features_dict['latitude']} (type: {type(features_dict['latitude'])})")
        
    #     # Create numpy array with explicit dtype
    #     features_array = np.array([[
    #         float(features_dict['ndvi']),
    #         float(features_dict['ndwi']),
    #         float(features_dict['cloud_cover']),
    #         float(features_dict['latitude'])
    #     ]], dtype=np.float64)
        
    #     print(f"   Features array: {features_array}")
    #     print(f"   Array shape: {features_array.shape}")
    #     print(f"   Array dtype: {features_array.dtype}")
        
    #     # Check scaler stats
    #     print(f"   Scaler mean: {SCALER.mean_ if hasattr(SCALER, 'mean_') else 'N/A'}")
    #     print(f"   Scaler scale: {SCALER.scale_ if hasattr(SCALER, 'scale_') else 'N/A'}")
        
    #     # Scale features
    #     features_scaled = SCALER.transform(features_array)
    #     print(f"   Scaled features: {features_scaled}")
        
    #     # Get prediction probabilities
    #     probabilities = ML_MODEL.predict_proba(features_scaled)
    #     print(f"   Raw probabilities: {probabilities}")
        
    #     # Handle different model output formats
    #     if probabilities.shape[1] == 2:  # Binary classification
    #         bloom_probability = probabilities[0, 1]  # Probability of class 1 (bloom)
    #     else:
    #         bloom_probability = probabilities[0, 0]  # Single class or different format
        
    #     prediction = ML_MODEL.predict(features_scaled)[0]
        
    #     print(f"   Final bloom probability: {bloom_probability:.4f} ({bloom_probability*100:.2f}%)")
    #     print(f"   Prediction class: {prediction}")
        
    #     # Determine confidence
    #     if bloom_probability > 0.75 or bloom_probability < 0.25:
    #         confidence = 'high'
    #     elif bloom_probability > 0.55 or bloom_probability < 0.45:
    #         confidence = 'medium'
    #     else:
    #         confidence = 'low'
        
    #     return {
    #         'bloom_probability': round(float(bloom_probability * 100), 2),
    #         'prediction': 'BLOOM' if prediction == 1 else 'NO_BLOOM',
    #         'confidence': confidence,
    #     }
        
    # except Exception as e:
    #     print(f"❌ ML prediction error: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return predict_bloom_fallback(features_dict)
    return predict_bloom_fallback(features_dict) 

def predict_bloom_fallback(features_dict):
    """Fallback heuristic if ML model fails"""
    ndvi = features_dict['ndvi']
    ndwi = features_dict['ndwi']
    
    score = 0.0
    
    if ndvi > 0.6:
        score += 60
    elif ndvi > 0.4:
        score += 40
    elif ndvi > 0.2:
        score += 20
    
    if -0.3 < ndwi < 0.1:
        score += 20
    elif -0.5 < ndwi < 0.3:
        score += 10
    
    probability = min(95, max(5, score))
    
    return {
        'bloom_probability': round(probability, 2),
        'prediction': 'BLOOM' if probability > 50 else 'NO_BLOOM',
        'confidence': 'low'
    }

# FastAPI Routes
@app.get("/")
async def root():
    return {
        "message": "Bloom Prediction API",
        "status": "active",
        "model_loaded": ML_MODEL is not None,
        "earth_engine": "initialized"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": ML_MODEL is not None}

@app.post("/predict", response_model=BloomPredictionResponse)
async def predict_bloom(request: BloomPredictionRequest):
    """
    Predict bloom probability for a given location and date
    
    - **lat**: Latitude (-90 to 90)
    - **lon**: Longitude (-180 to 180)  
    - **date**: Date in YYYY-MM-DD format
    """
    start_time = time.time()
    
    print(f"🌼 API Request: ({request.lat}, {request.lon}) on {request.date}")
    
    # Validate date format
    try:
        datetime.strptime(request.date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Get satellite data
    satellite_data = get_essential_vegetation_data(request.lat, request.lon, request.date)
    
    if not satellite_data:
        raise HTTPException(
            status_code=404,
            detail="No satellite data available for this location and time period (within 30 days)"
        )
    
    # Use ML model to predict bloom probability
    ml_prediction = predict_bloom_with_ml(satellite_data)
    
    # Build response
    prob = ml_prediction['bloom_probability']
    if prob > 70:
        message = '🌸 High likelihood of wildflower blooms!'
        recommendation = 'Great time to visit for wildflower viewing'
    elif prob > 40:
        message = '🌼 Moderate chance of wildflower blooms'
        recommendation = 'Blooms may be present but not peak conditions'
    else:
        message = '🍂 Low probability of blooms at this time'
        recommendation = 'Check back during peak bloom season (typically spring)'
    
    response = {
        'success': True,
        'bloom_probability': prob,
        'prediction': ml_prediction['prediction'],
        'confidence': ml_prediction['confidence'],
        'message': message,
        'analysis_date': satellite_data['actual_data_date'],
        'requested_date': satellite_data['original_request_date'],
        'data_quality': {
            'satellite': satellite_data['satellite'],
            'cloud_cover': round(satellite_data['cloud_cover'], 2),
            'days_offset': satellite_data['days_offset'],
            'buffer_radius_meters': satellite_data['buffer_size']
        },
        'vegetation_indices': {
            'ndvi': round(satellite_data['ndvi'], 4),
            'ndwi': round(satellite_data['ndwi'], 4),
            'ndvi_interpretation': (
                'High vegetation (bloom-like)' if satellite_data['ndvi'] > 0.6 
                else 'Moderate vegetation' if satellite_data['ndvi'] > 0.3 
                else 'Low vegetation (no bloom)' if satellite_data['ndvi'] > 0
                else 'Bare ground/water'
            )
        },
        'location': {
            'latitude': request.lat,
            'longitude': request.lon
        },
        'processing_time': round(time.time() - start_time, 2),
        'recommendation': recommendation
    }
    
    print(f"✅ API Response: {prob}% bloom probability")
    return response

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)