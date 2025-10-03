import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

print("=== BLOOM DETECTION MODEL TESTING ===\n")

# 1. Load the saved model and components
try:
    model = joblib.load('C:/Users/User/Desktop/nasashit/BloomAI/Training/v3/bloom_detection_model_fixed.joblib')
    scaler = joblib.load('C:/Users/User/Desktop/nasashit/BloomAI/Training/v3/feature_scaler_fixed.joblib')
    feature_columns = joblib.load('C:/Users/User/Desktop/nasashit/BloomAI/Training/v3/feature_columns_fixed.joblib')
    print("✅ Model and components loaded successfully")
    print(f"Feature columns: {feature_columns}")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    print("Please make sure these files exist in your current directory:")
    print("- bloom_detection_model_fixed.joblib")
    print("- feature_scaler_fixed.joblib") 
    print("- feature_columns_fixed.joblib")
    exit()

# 2. Create test dataset from your examples
test_data = [
    # bloom site blooming
    {
        'Site': 'San Carlos Reservation', 'year': 2020, 'NDVI': 0.1502692374398214,
        'NDWI': -0.1706359122161222, 'cloud_cover': 58.33, 
        'longitude': -110.30621859902435, 'latitude': 33.276734080468145,
        'bloom': 1, 'month': 3, 'day_of_year': 63, 'day_of_month': 3,
        'is_spring': 1, 'is_summer': 0, 'is_fall': 0, 'is_winter': 0
    },
    # non bloom site
    {
        'Site': 'Non_Bloom_1', 'year': 2020, 'NDVI': 0.1074567288389762,
        'NDWI': -0.1154643298932624, 'cloud_cover': 15.69,
        'longitude': -118.15401135088946, 'latitude': 33.89481895669643, 
        'bloom': 0, 'month': 1, 'day_of_year': 9, 'day_of_month': 9,
        'is_spring': 0, 'is_summer': 0, 'is_fall': 0, 'is_winter': 1
    },
    # bloom site non blooming season
    {
        'Site': 'North Table Mountain Ecological Reserve', 'year': 2023, 
        'NDVI': 0.2347942562700784, 'NDWI': -0.2287952210571442, 'cloud_cover': 0.47,
        'longitude': -121.55597836570155, 'latitude': 39.5934372299187,
        'bloom': 0, 'month': 1, 'day_of_year': 30, 'day_of_month': 30,
        'is_spring': 0, 'is_summer': 0, 'is_fall': 0, 'is_winter': 1
    },
    # non bloom site but spring
    {
        'Site': 'Non_Bloom_7', 'year': 2023, 'NDVI': 0.4603519222440965,
        'NDWI': -0.412935479921459, 'cloud_cover': 5.83,
        'longitude': -83.61273098446513, 'latitude': 35.62715942043859,
        'bloom': 0, 'month': 5, 'day_of_year': 144, 'day_of_month': 24,
        'is_spring': 1, 'is_summer': 0, 'is_fall': 0, 'is_winter': 0
    }
]

# Create DataFrame
test_df = pd.DataFrame(test_data)
print(f"\n✅ Test dataset created with {len(test_df)} samples")

# 3. Prepare features for prediction
X_test = test_df[feature_columns]
y_true = test_df['bloom']

print(f"\n📊 Test features shape: {X_test.shape}")
print("Test features summary:")
print(X_test.describe())

# 4. Scale features and make predictions
try:
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("✅ Predictions generated successfully")
except Exception as e:
    print(f"❌ Error during prediction: {e}")
    exit()

# 5. Display detailed results
print("\n" + "="*50)
print("📈 DETAILED TEST RESULTS")
print("="*50)

for i, (idx, row) in enumerate(test_df.iterrows()):
    actual_label = "BLOOM" if y_true.iloc[i] == 1 else "NO_BLOOM"
    predicted_label = "BLOOM" if y_pred[i] == 1 else "NO_BLOOM"
    probability = y_pred_proba[i]
    
    print(f"\n--- Test Case {i+1}: {row['Site']} ---")
    print(f"   Actual: {actual_label}")
    print(f"   Predicted: {predicted_label}")
    print(f"   Bloom Probability: {probability:.4f}")
    print(f"   Key Features -> NDVI: {row['NDVI']:.4f}, NDWI: {row['NDWI']:.4f}")
    print(f"   Confidence: {'HIGH' if probability > 0.7 or probability < 0.3 else 'MEDIUM' if probability > 0.55 or probability < 0.45 else 'LOW'}")
    
    # Check if prediction is correct
    if y_pred[i] == y_true.iloc[i]:
        print("   ✅ CORRECT PREDICTION")
    else:
        print("   ❌ MISCLASSIFICATION")

# 6. Overall test performance
print("\n" + "="*50)
print("📊 OVERALL TEST PERFORMANCE")
print("="*50)

accuracy = np.mean(y_pred == y_true)
print(f"Accuracy: {accuracy:.4f} ({np.sum(y_pred == y_true)}/{len(y_pred)} correct)")

try:
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")
except Exception as e:
    print(f"ROC-AUC: Could not calculate - {e}")

print(f"\nTrue Labels:      {list(y_true)}")
print(f"Predicted Labels: {list(y_pred)}")
print(f"Probabilities:    {[f'{p:.4f}' for p in y_pred_proba]}")

# 7. Feature analysis for insights
print("\n" + "="*50)
print("🔍 FEATURE ANALYSIS")
print("="*50)

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Model Feature Importance:")
print(feature_importance)

# 8. Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['NO_BLOOM', 'BLOOM'],
            yticklabels=['NO_BLOOM', 'BLOOM'])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Probability Distribution
colors = ['red' if actual == 1 else 'blue' for actual in y_true]
axes[1].bar(range(len(y_pred_proba)), y_pred_proba, color=colors, alpha=0.7)
axes[1].axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Test Case')
axes[1].set_ylabel('Bloom Probability')
axes[1].set_title('Prediction Probabilities\n(Red=Actual Bloom, Blue=Actual No Bloom)')
axes[1].set_xticks(range(len(y_pred_proba)))
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('model_testing_results.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'model_testing_results.png'")

# 9. Prediction function for new data
def predict_new_location(lat, ndvi, ndwi, cloud_cover, month, day_of_year):
    """
    Predict bloom probability for new location data
    """
    # Determine seasons
    is_spring = 1 if month in [3, 4, 5] else 0
    is_summer = 1 if month in [6, 7, 8] else 0
    is_fall = 1 if month in [9, 10, 11] else 0
    is_winter = 1 if month in [12, 1, 2] else 0
    
    features_dict = {
        'NDVI': ndvi,
        'NDWI': ndwi,
        'cloud_cover': cloud_cover,
        'latitude': lat
    }
    
    features_df = pd.DataFrame([features_dict])[feature_columns]
    features_scaled = scaler.transform(features_df)
    
    probability = model.predict_proba(features_scaled)[0, 1]
    prediction = model.predict(features_scaled)[0]
    
    if probability > 0.7 or probability < 0.3:
        confidence = 'HIGH'
    elif probability > 0.55 or probability < 0.45:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    return {
        'bloom_probability': round(probability, 4),
        'prediction': 'BLOOM' if prediction == 1 else 'NO_BLOOM',
        'confidence': confidence
    }

# 10. Test the prediction function with your examples
print("\n" + "="*50)
print("🎯 TESTING PREDICTION FUNCTION")
print("="*50)

test_cases = [
    (33.276, 0.15, -0.17, 58.33, 3, 63),   # San Carlos
    (33.895, 0.107, -0.115, 15.69, 1, 9),  # Non_Bloom_1
]

for i, (lat, ndvi, ndwi, cloud, month, doy) in enumerate(test_cases):
    result = predict_new_location(lat, ndvi, ndwi, cloud, month, doy)
    print(f"Test {i+1}: {result}")

print("\n" + "="*50)
print("✅ TESTING COMPLETE")
print("="*50)
print("Check the generated 'model_testing_results.png' for visual analysis")