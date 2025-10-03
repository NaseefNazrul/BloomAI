import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("=== BLOOM DETECTION MODEL TRAINING (FIXED) ===\n")

# 1. Load Dataset
df = pd.read_csv('C:/Users/User/Desktop/nasashit/BloomAI/V2/training/training_dataset.csv')

print(f"Original dataset shape: {df.shape}")
print(f"Original target distribution:\n{df['bloom'].value_counts()}")

# 2. Analyze and clean data
high_confidence_blooms = df[(df['bloom'] == 1) & (df['NDVI'] > 0.25)]
print(f"High-confidence blooms: {len(high_confidence_blooms)}")

# Remove ambiguous cases - blooms with very low NDVI (probably mislabeled)
clean_df = df[~((df['bloom'] == 1) & (df['NDVI'] < 0.15))].copy()
clean_df.reset_index(drop=True, inplace=True)  # CRITICAL: Reset index after filtering
print(f"After cleaning ambiguous blooms: {clean_df.shape}")
print(f"Cleaned target distribution:\n{clean_df['bloom'].value_counts()}")

# 3. Feature selection
feature_columns = [
    'NDVI',           # Vegetation index - MOST IMPORTANT
    'NDWI',           # Water index
    'cloud_cover',    # Cloud coverage
    'latitude',       # Location matters for climate
]

X = clean_df[feature_columns].copy()
y = clean_df['bloom'].copy()

print(f"\nFeatures used: {feature_columns}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# 4. Check data distribution
print("\n=== DATA QUALITY CHECKS ===")
print("NDVI range:", X['NDVI'].min(), "to", X['NDVI'].max())
print("NDWI range:", X['NDWI'].min(), "to", X['NDWI'].max())
print("\nNDVI by bloom status (CLEANED DATA):")
print(clean_df.groupby('bloom')['NDVI'].describe())

# 5. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
X_scaled.reset_index(drop=True, inplace=True)  # Reset index for alignment

# 6. Handle Class Imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"\nClass weights: {class_weight_dict}")

# 7. Split Data - Use site-based split to prevent leakage
unique_sites = clean_df['Site'].unique()
np.random.seed(42)
train_sites = np.random.choice(unique_sites, size=int(0.8 * len(unique_sites)), replace=False)

train_mask = clean_df['Site'].isin(train_sites)

# FIX: Reset indices to ensure alignment
X_train = X_scaled.loc[train_mask].reset_index(drop=True)
X_test = X_scaled.loc[~train_mask].reset_index(drop=True)
y_train = y[train_mask].reset_index(drop=True)
y_test = y[~train_mask].reset_index(drop=True)

print(f"\nTraining sites: {len(train_sites)}")
print(f"Test sites: {len(unique_sites) - len(train_sites)}")
print(f"Training samples: {X_train.shape}, {y_train.shape}")
print(f"Test samples: {X_test.shape}, {y_test.shape}")

# 8. Train Model
print("\n=== TRAINING MODEL ===")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight=class_weight_dict,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 9. Evaluate Model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n=== MODEL PERFORMANCE ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
try:
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
except ValueError as e:
    print(f"ROC-AUC: Could not calculate (only one class in test set)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Bloom', 'Bloom']))

# 10. Feature Importance Analysis
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance)

# 11. Cross-Validation
print("\n=== CROSS-VALIDATION ===")
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 12. Confusion Matrix Analysis
cm = confusion_matrix(y_test, y_pred)
print("\n=== CONFUSION MATRIX ===")
print(cm)
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# 13. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# Feature Importance
sns.barplot(data=feature_importance, x='importance', y='feature', ax=axes[0, 1], hue='feature', legend=False)
axes[0, 1].set_title('Feature Importance')

# NDVI Distribution by Bloom Status (CLEANED DATA)
clean_df.boxplot(column='NDVI', by='bloom', ax=axes[1, 0])
axes[1, 0].set_title('NDVI Distribution by Bloom Status (Cleaned Data)')
axes[1, 0].set_xlabel('Bloom (0=No, 1=Yes)')
axes[1, 0].set_ylabel('NDVI')

# Prediction Probability Distribution
if len(np.unique(y_test)) > 1:
    axes[1, 1].hist([y_pred_proba[y_test == 0], y_pred_proba[y_test == 1]], 
                    bins=20, label=['No Bloom', 'Bloom'], alpha=0.7)
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
else:
    axes[1, 1].text(0.5, 0.5, 'Only one class in test set', 
                    ha='center', va='center', transform=axes[1, 1].transAxes)

plt.tight_layout()
plt.savefig('bloom_model_evaluation_fixed.png', dpi=300, bbox_inches='tight')
print("\n=== Visualizations saved ===")
plt.show()

# 14. Save Model
joblib.dump(model, 'bloom_detection_model_fixed.joblib')
joblib.dump(scaler, 'feature_scaler_fixed.joblib')
joblib.dump(feature_columns, 'feature_columns_fixed.joblib')

print("\n=== MODEL SAVED ===")

# 15. Prediction Function
def predict_bloom(lat, ndvi, ndwi, cloud_cover):
    """
    Predict bloom probability based on satellite indices
    
    Args:
        lat: Latitude
        ndvi: Normalized Difference Vegetation Index
        ndwi: Normalized Difference Water Index
        cloud_cover: Cloud cover percentage
    
    Returns:
        dict with bloom_probability, prediction, and confidence
    """
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

# 16. Test Prediction
print("\n=== TESTING PREDICTION ===")
test_pred = predict_bloom(
    lat=34.7,
    ndvi=0.35,
    ndwi=-0.15,
    cloud_cover=10.0
)
print(f"Test prediction: {test_pred}")

print("\n=== IMPORTANT NOTES ===")
print("1. Removed ambiguous blooms (NDVI < 0.15) to improve signal quality")
print("2. Using site-based splitting to prevent overfitting")
print("3. Focused on satellite-derived features only (no temporal leakage)")
print("4. Model should now be more reliable for real-world predictions")
print("\n=== TRAINING COMPLETE ===")