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
    # Startup: Load MIL model and initialize Earth Engine
    global ML_MODEL, SCALER, FEATURE_COLUMNS
    
    try:
        ML_MODEL = joblib.load('C:/Users/User/Desktop/nasashit/BloomAI/App/mil_bloom_model.joblib')
        SCALER = joblib.load('C:/Users/User/Desktop/nasashit/BloomAI/App/mil_scaler.joblib')
        FEATURE_COLUMNS = joblib.load('C:/Users/User/Desktop/nasashit/BloomAI/App/mil_features.joblib')
        print("✅ MIL Model loaded successfully")
        print(f"✅ Features: {FEATURE_COLUMNS}")
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
    description="Predict wildflower bloom probability using satellite data and MIL model",
    version="2.0.0",
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
    Get essential vegetation data optimized for MIL bloom prediction
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
    """Get vegetation data for a single date and satellite - NOW WITH EVI AND LST"""
    
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
        
        # Calculate ALL essential indices for MIL model
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
        
        # Enhanced Vegetation Index (EVI) - important for MIL model
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('SR_B5'),
                'RED': image.select('SR_B4'),
                'BLUE': image.select('SR_B2')
            }
        ).rename('EVI')
        
        # Land Surface Temperature (LST) - important for MIL model
        lst = image.select('ST_B10')\
            .multiply(0.00341802)\
            .add(149.0)\
            .subtract(273.15)\
            .rename('LST')
        
        # Combine all bands for single API call
        composite = ndvi.addBands(ndwi).addBands(evi).addBands(lst)
        
        # Single efficient API call for all indices
        area_stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=area,
            scale=30,
            maxPixels=1e6,
            bestEffort=True
        ).getInfo()
        
        ndvi_val = area_stats.get('NDVI')
        ndwi_val = area_stats.get('NDWI')
        evi_val = area_stats.get('EVI')
        lst_val = area_stats.get('LST')
        
        if ndvi_val is None:
            return None
        
        # Get month and day_of_year for MIL model features
        current_dt = datetime.strptime(date, '%Y-%m-%d')
        month = current_dt.month
        day_of_year = current_dt.timetuple().tm_yday
        
        return {
            'ndvi': float(ndvi_val),
            'ndwi': float(ndwi_val),
            'evi': float(evi_val),
            'lst': float(lst_val),
            'cloud_cover': float(cloud_cover),
            'month': month,
            'day_of_year': day_of_year,
            'satellite': satellite,
            'date': date,
            'buffer_size': buffer_meters
        }
        
    except Exception as e:
        print(f"   Error getting {satellite} data for {date}: {e}")
        return None

def predict_bloom_with_ml(features_dict):
    """
    FINAL TWEAK: Better balance for bloom sites while keeping non-bloom protection
    """
    # Extract key features
    ndvi = features_dict['ndvi']
    evi = features_dict['evi']
    ndwi = features_dict['ndwi']
    lst = features_dict['lst']
    month = features_dict['month']

    if month in [11, 12, 1, 2]:  # Winter months
        if evi < 0.8 and ndvi < 0.3:  # Not extremely high vegetation
            print("❄️  WINTER ADJUSTMENT: Reducing probability for winter season")
            # We'll apply this adjustment later after ML prediction
    
    # ONLY override for clearly impossible conditions
    if ndvi < 0.05:  # Bare ground/water
        print("⚠️  OVERRIDE: Extremely low NDVI - forcing NO_BLOOM")
        return {
            'bloom_probability': 8.0,
            'prediction': 'NO_BLOOM',
            'confidence': 'HIGH'
        }
    
    if evi < 0.1 and ndvi < 0.1:  # Both indices extremely low
        print("⚠️  OVERRIDE: Extremely low vegetation - forcing NO_BLOOM")
        return {
            'bloom_probability': 10.0,
            'prediction': 'NO_BLOOM', 
            'confidence': 'HIGH'
        }
    
    # Only use ML model if not in extreme cases
    if ML_MODEL is not None:
        try:
            features_array = np.array([[
                float(features_dict['ndvi']),
                float(features_dict['ndwi']), 
                float(features_dict['evi']),
                float(features_dict['lst']),
                float(features_dict['cloud_cover']),
                float(features_dict['month']),
                float(features_dict['day_of_year'])
            ]], dtype=np.float64)
            
            features_scaled = SCALER.transform(features_array)
            probabilities = ML_MODEL.predict_proba(features_scaled)
            
            if probabilities.shape[1] == 2:
                bloom_probability = probabilities[0, 1]
            else:
                bloom_probability = probabilities[0, 0]
            
            prediction = ML_MODEL.predict(features_scaled)[0]

            if month in [11, 12, 1, 2] and evi < 0.8 and ndvi < 0.3:
                winter_factor = 0.5  # Reduce probability by 50% in winter
                bloom_probability = bloom_probability * winter_factor
                print(f"❄️  Applied winter adjustment (factor: {winter_factor})")
            
            # IMPROVED ADJUSTMENT: Be more generous for bloom sites with decent EVI
            if bloom_probability > 0.8:
                if evi > 0.5:  # Good EVI - less penalty
                    adjustment_factor = min(1.0, max(0.7, ndvi / 0.25))  # Minimum 0.7 factor if EVI is good
                    bloom_probability = bloom_probability * adjustment_factor
                    print(f"⚠️  LIGHT ADJUSTMENT: Good EVI, reduced penalty (factor: {adjustment_factor:.2f})")
                elif ndvi < 0.25:  # Moderate NDVI - moderate penalty
                    adjustment_factor = min(1.0, ndvi / 0.3)
                    bloom_probability = bloom_probability * adjustment_factor
                    print(f"⚠️  ADJUSTED: Scaled high probability due to moderate vegetation (factor: {adjustment_factor:.2f})")
            
            # ENHANCED Seasonal confidence boost for spring months
            if month in [3, 4, 5] and bloom_probability > 0.3:
                # More generous boost for spring when blooms are more likely
                boost_factor = 1.15 if evi > 0.4 else 1.05
                bloom_probability = min(0.95, bloom_probability * boost_factor)
                print(f"✅ Seasonal boost applied for spring months (factor: {boost_factor})")
            
            # BETTER Confidence calculation
            if bloom_probability > 0.75 or bloom_probability < 0.25:
                confidence = 'HIGH'
            elif bloom_probability > 0.6 or bloom_probability < 0.4:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            return {
                'bloom_probability': round(float(bloom_probability * 100), 2),
                'prediction': 'BLOOM' if prediction == 1 else 'NO_BLOOM',
                'confidence': confidence,
            }
            
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
    
    # Enhanced fallback
    return predict_bloom_fallback(features_dict)

def predict_bloom_fallback(features_dict):
    """Optimized fallback heuristic for better bloom site detection"""
    ndvi = features_dict['ndvi']
    ndwi = features_dict['ndwi']
    evi = features_dict['evi']
    lst = features_dict['lst']
    month = features_dict['month']
    
    score = 0.0
    
    # IMPROVED: More emphasis on EVI for bloom sites
    if evi > 0.7:
        score += 55  # Increased from 50
    elif evi > 0.5:
        score += 40  # Increased from 35
    elif evi > 0.3:
        score += 25  # Increased from 20
    elif evi > 0.15:
        score += 8   # Increased from 5
    
    # IMPROVED: Better NDVI scoring
    if ndvi > 0.5:
        score += 30  # Increased from 25
    elif ndvi > 0.3:
        score += 20  # Increased from 15
    elif ndvi > 0.15:
        score += 8   # Increased from 5
    
    # IMPROVED: Environmental factors with better thresholds
    if -0.2 < ndwi < 0.05:
        score += 15
    
    # Better temperature range for blooms
    if 12 < lst < 32:  # Wider optimal range
        score += 12
    elif 8 < lst < 38:  # Extended acceptable range
        score += 5
    
    # IMPROVED: Stronger seasonal adjustment for spring
    if month in [3, 4, 5]:  # Spring
        score += 15  # Increased from 10
    elif month in [11, 12, 1, 2]:  # Winter
        score -= 3   # Reduced from 5 (less penalty)
    
    # Cap the probability
    probability = min(90, max(8, score))
    
    # IMPROVED: More nuanced prediction thresholds
    if probability > 52:  # Lowered from 55
        prediction = 'BLOOM'
        confidence = 'MEDIUM' if probability > 65 else 'LOW'
    else:
        prediction = 'NO_BLOOM' 
        confidence = 'MEDIUM' if probability < 25 else 'LOW'
    
    return {
        'bloom_probability': round(probability, 2),
        'prediction': prediction,
        'confidence': confidence
    }

# FastAPI Routes
@app.get("/")
async def root():
    return {
        "message": "Bloom Prediction API with MIL Model",
        "status": "active",
        "model_loaded": ML_MODEL is not None,
        "model_type": "MIL (Multiple Instance Learning)",
        "features": FEATURE_COLUMNS if FEATURE_COLUMNS else "Not loaded"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": ML_MODEL is not None,
        "earth_engine": "initialized"
    }

@app.post("/predict", response_model=BloomPredictionResponse)
async def predict_bloom(request: BloomPredictionRequest):
    """
    Predict bloom probability for a given location and date using MIL model
    
    - **lat**: Latitude (-90 to 90)
    - **lon**: Longitude (-180 to 180)  
    - **date**: Date in YYYY-MM-DD format
    """
    start_time = time.time()
    
    print(f"🌼 MIL API Request: ({request.lat}, {request.lon}) on {request.date}")
    
    # Validate date format
    try:
        datetime.strptime(request.date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Get satellite data with ALL features needed for MIL model
    satellite_data = get_essential_vegetation_data(request.lat, request.lon, request.date)
    
    if not satellite_data:
        raise HTTPException(
            status_code=404,
            detail="No satellite data available for this location and time period (within 30 days)"
        )
    
    # Use MIL model to predict bloom probability
    ml_prediction = predict_bloom_with_ml(satellite_data)
    
    # Build response
    prob = ml_prediction['bloom_probability']
    
    # Enhanced messaging based on MIL model confidence
    if prob > 80:
        message = '🌸 HIGH BLOOM PROBABILITY - Excellent conditions detected!'
        recommendation = 'Prime time for wildflower viewing based on vegetation patterns'
    elif prob > 60:
        message = '🌼 MODERATE BLOOM PROBABILITY - Good conditions'
        recommendation = 'Likely blooms present, good time to visit'
    elif prob > 40:
        message = '🌱 UNCERTAIN BLOOM PROBABILITY - Mixed signals'
        recommendation = 'Some vegetation activity detected, but not peak conditions'
    else:
        message = '🍂 LOW BLOOM PROBABILITY - Minimal activity'
        recommendation = 'Check back during optimal seasonal periods'
    
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
            'evi': round(satellite_data['evi'], 4),  # NEW: EVI for MIL model
            'lst': round(satellite_data['lst'], 2),   # NEW: LST for MIL model
            'mil_model_features': FEATURE_COLUMNS if FEATURE_COLUMNS else []
        },
        'location': {
            'latitude': request.lat,
            'longitude': request.lon
        },
        'processing_time': round(time.time() - start_time, 2),
        'recommendation': recommendation
    }
    
    print(f"✅ MIL API Response: {prob}% bloom probability ({ml_prediction['confidence']} confidence)")
    return response

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)