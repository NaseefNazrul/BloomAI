# Bloom Prediction API — Endpoints Cheat Sheet

Bloom Ml model with API endpoints, able to detect blooms depending on the locations 

**Base URL (local dev)**: `http://localhost:8000`  
**Default port used in example app**: `8000`

---

# Endpoints

## `GET /`
Simple service info

- **URL**: `/`
- **Method**: `GET`
- **Purpose**: Quick sanity check & basic status
- **Response (200)**:
```json
{
  "message": "Bloom Prediction API",
  "status": "active",
  "model_loaded": true,
  "earth_engine": "initialized"
}
```
- `model_loaded` indicates if the on-disk ML model loaded successfully.
- `earth_engine` shows whether Earth Engine init succeeded.

---

## `GET /health`
Health check for monitoring

- **URL**: `/health`
- **Method**: `GET`
- **Purpose**: Liveness/readiness probe
- **Response (200)**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## `POST /predict`
Predict bloom probability for a location + date.

- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body schema** (required fields):

```json
{
  "lat": 33.2767,
  "lon": -110.3062,
  "date": "2023-04-15"
}
```

**Field rules**
- `lat`: float — latitude in degrees, between `-90` and `90`.
- `lon`: float — longitude in degrees, between `-180` and `180`.
- `date`: string — date in `YYYY-MM-DD` format.

**Success response (200)** — `BloomPredictionResponse`:
```json
{
  "success": true,
  "bloom_probability": 72.45,
  "prediction": "BLOOM",
  "confidence": "low",
  "message": "🌸 High likelihood of wildflower blooms!",
  "analysis_date": "2023-04-13",
  "requested_date": "2023-04-15",
  "data_quality": {
    "satellite": "Landsat-9",
    "cloud_cover": 12.34,
    "days_offset": 2,
    "buffer_radius_meters": 200
  },
  "vegetation_indices": {
    "ndvi": 0.6213,
    "ndwi": -0.0456,
    "ndvi_interpretation": "High vegetation (bloom-like)"
  },
  "location": {
    "latitude": 33.2767,
    "longitude": -110.3062
  },
  "processing_time": 2.34,
  "recommendation": "Great time to visit for wildflower viewing"
}
```

**Meaning of important fields**
- `bloom_probability`: percentage (0–100) estimated chance of bloom (from model or fallback heuristic).
- `prediction`: `"BLOOM"` or `"NO_BLOOM"` (classification).
- `confidence`: qualitative confidence: `"low"`, `"medium"`, or `"high"`.
- `analysis_date`: actual satellite image date used (may differ from `requested_date` due to fallback).
- `requested_date`: date the user asked for.
- `data_quality`: metadata about the satellite data and any fallback:
  - `days_offset`: how many days earlier the used image is vs requested date (0 = same day).
  - `cloud_cover`: percent cloud over the selected image.
  - `buffer_radius_meters`: radius used around the point to compute indices.
- `vegetation_indices`: numeric indices used by the model (NDVI/NDWI) and a short interpretation.
- `processing_time`: seconds it took for the backend to process the request and return a response.
- `recommendation`: short user-facing guidance based on probability ranges.

---

# Errors & status codes

- **400 Bad Request**
  - Invalid request body or bad date format.
  - Example: `{"detail":"Invalid date format. Use YYYY-MM-DD"}`

- **404 Not Found**
  - No satellite data available within the configured fallback window (default 30 days).
  - Example: `{"detail":"No satellite data available for this location and time period (within 30 days)"}`

- **500 Internal Server Error**
  - Earth Engine initialization or internal processing error.
  - Example: `{"detail":"Earth Engine error: <message>"}`

- **502 / 504 Gateway errors**
  - If external model backend or third-party inference times out or returns an error (applies if integrated with remote model service).

---

# Quick examples

### curl — local predict
```bash
curl -X POST "http://localhost:8000/predict"   -H "Content-Type: application/json"   -d '{"lat": 33.2767, "lon": -110.3062, "date": "2023-04-15"}'
```

### Python (requests)
```python
import requests

payload = {"lat": 33.2767, "lon": -110.3062, "date": "2023-04-15"}
resp = requests.post("http://localhost:8000/predict", json=payload, timeout=60)
resp.raise_for_status()
data = resp.json()
print(data)
```

### JavaScript (fetch)
```javascript
const payload = { lat: 33.2767, lon: -110.3062, date: "2023-04-15" };
fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
})
  .then(res => res.json())
  .then(json => console.log(json))
  .catch(err => console.error("API error:", err));
```

---

# Integration tips for frontend developer

- **Timeouts:** Earth Engine calls + model inference can be slow. Set client-side timeouts to ~60s for initial development; implement a loading state in UI.
- **Cold starts:** If the model or external model service sleeps, the first call may be slow. Consider a warm-up request on deploy.
- **Cache repeated queries:** If many users query the same `lat/lon/date`, cache results for a short time to reduce Earth Engine and model load.
- **Validation:** Validate `lat`, `lon`, and `date` on the frontend before calling the API to reduce bad requests.
- **Handle partial data:** The API may return `days_offset` > 0 when it uses earlier satellite imagery — display `analysis_date` vs `requested_date` to the user.
- **Rate limiting & abuse:** If deployed publicly, add rate-limiting on the backend to avoid excessive Earth Engine usage or third-party charges.

---

# Environment & notes (for local dev)
- App runs on `uvicorn` by default: `uvicorn app:app --host 0.0.0.0 --port 8000`
- Ensure Earth Engine is initialized for the user/service account running the backend.
- If the ML model file(s) are absent or fail to load, the API uses a fallback heuristic. `GET /` and `GET /health` show `model_loaded` status.

---

# Contact / next steps
If you want, I can:
- Add example Postman collection JSON for these endpoints, or
- Provide a small example React component that calls `/predict` and displays `vegetation_indices` and `recommendation`.

Let me know which one you want and I’ll drop it into the repo README.
