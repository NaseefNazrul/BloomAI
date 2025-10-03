import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=== MULTIPLE INSTANCE LEARNING FOR BLOOM DETECTION ===")

# Load your training CSV
df = pd.read_csv('C:/Users/User/Desktop/nasashit/BloomAI/V2/training/training_dataset.csv')  # Replace with your actual file path

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Bloom site distribution:\n{df['bloom'].value_counts()}")

# Check data quality
print(f"\n=== DATA QUALITY CHECK ===")
print(f"Number of unique sites: {df['Site'].nunique()}")
print(f"Missing values:\n{df.isnull().sum()}")

# Remove any rows with missing critical features
df_clean = df.dropna(subset=['NDVI', 'NDWI', 'EVI', 'LST'])
print(f"After cleaning: {df_clean.shape}")

# Define features - using vegetation indices, climate, and temporal features
feature_columns = ['NDVI', 'NDWI', 'EVI', 'LST', 'cloud_cover', 'month', 'day_of_year']

print(f"\n=== MIL DATA PREPARATION ===")
print(f"Using features: {feature_columns}")

# Prepare data in MIL format
def prepare_mil_data(df):
    """
    Convert dataframe to MIL format where:
    - Each Site is a 'bag'
    - Each row for that Site is an 'instance'
    - Bag label = bloom (1 for bloom sites, 0 for non-bloom sites)
    """
    bags = []
    bag_labels = []
    site_ids = []
    
    for site_name, group in df.groupby('Site'):
        # Each bag contains multiple instances (time-series observations)
        bag_instances = group[feature_columns].values
        bags.append(bag_instances)
        
        # Bag label is the bloom status (should be same for all instances of the site)
        bag_label = group['bloom'].iloc[0]
        bag_labels.append(bag_label)
        site_ids.append(site_name)
    
    return bags, bag_labels, site_ids

# Convert to MIL format
bags, bag_labels, site_ids = prepare_mil_data(df_clean)

print(f"Number of bags (sites): {len(bags)}")
print(f"Bag labels - Bloom sites: {sum(bag_labels)}, Non-bloom sites: {len(bag_labels) - sum(bag_labels)}")
print(f"Instances per bag statistics:")
instance_counts = [len(bag) for bag in bags]
print(f"  Min: {min(instance_counts)}, Max: {max(instance_counts)}, Mean: {np.mean(instance_counts):.1f}")

# MIL Cross-Validation with GroupKFold (respects bag boundaries)
def mil_cross_validation(bags, bag_labels, site_ids, n_splits=5):
    """Perform MIL cross-validation ensuring instances from same site stay together"""
    
    # Create flat arrays for group K-Fold
    X_flat = np.vstack(bags)
    y_flat = np.concatenate([[label] * len(bag) for label, bag in zip(bag_labels, bags)])
    groups = np.concatenate([[i] * len(bag) for i, bag in enumerate(bags)])
    
    group_kfold = GroupKFold(n_splits=n_splits)
    
    cv_scores = []
    feature_importances = []
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_idx, test_idx) in enumerate(group_kfold.split(X_flat, y_flat, groups), 1):
        print(f"  Fold {fold}: Training on {len(train_idx)} instances, Testing on {len(test_idx)} instances")
        
        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y_flat[train_idx], y_flat[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=0.6,
            random_state=42,
            class_weight={0: 24, 1: 1},  # Manual weighting for the 24:1 imbalance
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        feature_importances.append(model.feature_importances_)
        
        # Bag-level prediction for test set
        test_bag_indices = np.unique(groups[test_idx])
        bag_predictions = []
        bag_truths = []
        
        for bag_idx in test_bag_indices:
            # Get all instances for this bag
            bag_instance_mask = (groups == bag_idx)
            bag_features = X_flat[bag_instance_mask]
            bag_features_scaled = scaler.transform(bag_features)  # Scale using same scaler
            
            # Get instance probabilities
            instance_proba_result = model.predict_proba(bag_features_scaled)
            # Check the number of classes predicted and get probability for the positive class if possible
            if instance_proba_result.shape[1] == 2:
                instance_probs = instance_proba_result[:, 1]  # Positive class probability
            else:
                # If only one class is predicted, use zeros or the available class probability
                # This happens when model only predicts one class
                print(f"Warning: Model predicting only {instance_proba_result.shape[1]} class(es)")
                instance_probs = instance_proba_result[:, 0]
            
            # MIL aggregation: Use maximum probability in bag
            bag_prob = np.max(instance_probs)
            bag_pred = 1 if bag_prob > 0.5 else 0
            
            bag_predictions.append(bag_pred)
            bag_truths.append(bag_labels[bag_idx])
            
            # Store for overall evaluation
            all_y_true.extend([bag_labels[bag_idx]])
            all_y_pred.extend([bag_pred])
        
        # Bag-level accuracy for this fold
        fold_accuracy = accuracy_score(bag_truths, bag_predictions)
        cv_scores.append(fold_accuracy)
        print(f"    Fold {fold} Bag-level Accuracy: {fold_accuracy:.4f}")
    
    return cv_scores, feature_importances, all_y_true, all_y_pred

print("\n=== MIL CROSS-VALIDATION ===")
cv_scores, feature_importances, all_y_true, all_y_pred = mil_cross_validation(bags, bag_labels, site_ids)

print(f"\nOverall MIL Performance:")
print(f"Bag-level CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

# Final model training on all data
print("\n=== TRAINING FINAL MIL MODEL ===")

# Prepare all data for final training
X_all = np.vstack(bags)
y_all = np.concatenate([[label] * len(bag) for label, bag in zip(bag_labels, bags)])

# Scale features
final_scaler = StandardScaler()
X_all_scaled = final_scaler.fit_transform(X_all)

# Train final model
final_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=8,
    min_samples_leaf=3,
    max_features=0.6,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

final_model.fit(X_all_scaled, y_all)

# Feature importance
avg_feature_importance = np.mean(feature_importances, axis=0)
feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': avg_feature_importance
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance_df)

# MIL Prediction function for new sites
def predict_site_bloom_mil(site_observations, model, scaler, feature_columns):
    """
    Predict bloom for a new site using MIL approach
    
    Parameters:
    - site_observations: DataFrame with multiple observations from the same site
    - Must contain the feature_columns
    
    Returns:
    - Dictionary with bloom prediction and confidence
    """
    if len(site_observations) == 0:
        return {'bloom_probability': 0.0, 'prediction': 'NO_BLOOM', 'confidence': 'LOW'}
    
    # Extract features
    X_site = site_observations[feature_columns].values
    
    # Scale features
    X_site_scaled = scaler.transform(X_site)
    
    # Get instance probabilities
    instance_probs = model.predict_proba(X_site_scaled)[:, 1]
    
    # MIL aggregation: Use maximum probability
    max_prob = np.max(instance_probs)
    mean_prob = np.mean(instance_probs)
    
    # Count instances with high bloom probability
    high_prob_instances = np.sum(instance_probs > 0.7)
    total_instances = len(instance_probs)
    
    # Determine prediction and confidence
    if max_prob > 0.6:  # Conservative threshold
        prediction = 'BLOOM'
        if max_prob > 0.8 and high_prob_instances > total_instances * 0.3:
            confidence = 'HIGH'
        elif max_prob > 0.7:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
    else:
        prediction = 'NO_BLOOM'
        if max_prob < 0.3:
            confidence = 'HIGH'
        else:
            confidence = 'LOW'
    
    return {
        'bloom_probability': float(max_prob),
        'mean_probability': float(mean_prob),
        'prediction': prediction,
        'confidence': confidence,
        'high_prob_instances': int(high_prob_instances),
        'total_instances': int(total_instances)
    }

# Test the MIL model with realistic scenarios
print("\n=== MIL PREDICTION TESTING ===")

# Create test scenarios that mimic real site data
test_scenarios = {
    'Non_Bloom_Summer_2016': pd.DataFrame([{
        'NDVI': 0.0873491841377194, 'NDWI': -0.1282278024527789, 'EVI': 0.531658553982041, 
        'LST': 50.665353315729085, 'cloud_cover': 6.66, 'month': 8, 'day_of_year': 223
    }]),
    
    'San_Carlos_Winter_2019': pd.DataFrame([{
        'NDVI': 0.126283997983929, 'NDWI': -0.1548695949290675, 'EVI': 0.9462388221069744, 
        'LST': 11.794194783938137, 'cloud_cover': 8.09, 'month': 1, 'day_of_year': 11
    }]),
    
    'Suguaro_Winter_2017': pd.DataFrame([{
        'NDVI': 0.1424686656685659, 'NDWI': -0.1650192961265499, 'EVI': 0.8565993677154015, 
        'LST': 18.449970704424253, 'cloud_cover': 1.94, 'month': 1, 'day_of_year': 28
    }]),
    
    'Carrizo_Fall_2022': pd.DataFrame([{
        'NDVI': 0.1100976820034903, 'NDWI': -0.1521724694959364, 'EVI': 0.7713496064791002, 
        'LST': 17.605599368514856, 'cloud_cover': 15.44, 'month': 11, 'day_of_year': 316
    }]),
    
    'Non_Bloom_Summer_2024': pd.DataFrame([{
        'NDVI': 0.0798993881749185, 'NDWI': -0.1214178676048647, 'EVI': 0.4593794464784901, 
        'LST': 54.07187598464823, 'cloud_cover': 0.0, 'month': 6, 'day_of_year': 165
    }]),
    
    'Non_Bloom_Winter_2024': pd.DataFrame([{
        'NDVI': 0.2028752378914867, 'NDWI': -0.2001461983446526, 'EVI': 1.6364287060306564, 
        'LST': 4.3151954395183285, 'cloud_cover': 7.79, 'month': 12, 'day_of_year': 354
    }])
}

print("Testing MIL predictions on different site patterns:\n")
for scenario_name, observations in test_scenarios.items():
    result = predict_site_bloom_mil(observations, final_model, final_scaler, feature_columns)
    print(f"{scenario_name:25} | "
          f"Max Prob: {result['bloom_probability']:5.3f} | "
          f"Pred: {result['prediction']:8} | "
          f"Conf: {result['confidence']:6} | "
          f"HighProb: {result['high_prob_instances']}/{result['total_instances']}")

# Save the final model and scaler
import joblib
joblib.dump(final_model, 'mil_bloom_model.joblib')
joblib.dump(final_scaler, 'mil_scaler.joblib')
joblib.dump(feature_columns, 'mil_features.joblib')

print(f"\n=== MIL MODEL TRAINING COMPLETE ===")
print(f"Final model saved with {len(feature_columns)} features")
print(f"Average cross-validation accuracy: {np.mean(cv_scores):.1%}")
print(f"Most important feature: {feature_importance_df.iloc[0]['feature']}")

# Check if the model is learning vegetation patterns
top_features = feature_importance_df.head(3)['feature'].tolist()
vegetation_features = ['NDVI', 'NDWI', 'EVI']
if any(feat in vegetation_features for feat in top_features):
    print("✅ SUCCESS: Model is learning vegetation patterns")
else:
    print("⚠️  WARNING: Vegetation features not in top predictors")