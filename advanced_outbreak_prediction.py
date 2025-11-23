import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED DISEASE OUTBREAK PREDICTION MODEL")
print("Using Machine Learning Ensemble with Feature Engineering")
print("=" * 80)

# Load imputed data
print("\n1. Loading imputed data...")
df = pd.read_csv('combined_disease_data_imputed.csv')
print(f"   ✓ Loaded {len(df)} records")

# Get year columns
year_columns = [col for col in df.columns if str(col).isdigit()]
year_columns = sorted(year_columns, key=lambda x: int(x))

# Convert to numeric
for year in year_columns:
    df[year] = pd.to_numeric(df[year], errors='coerce')

print(f"   ✓ Year range: {year_columns[0]} - {year_columns[-1]}")

# Advanced Feature Engineering
print("\n2. Advanced Feature Engineering...")

def create_advanced_features(time_series, lookback=10):
    """Create comprehensive features from time series"""
    if len(time_series) < lookback:
        return None
    
    recent = time_series[-lookback:]
    
    features = {}
    
    # Statistical features
    features['mean'] = np.mean(recent)
    features['median'] = np.median(recent)
    features['std'] = np.std(recent)
    features['min'] = np.min(recent)
    features['max'] = np.max(recent)
    features['range'] = features['max'] - features['min']
    features['cv'] = features['std'] / (features['mean'] + 1)  # Coefficient of variation
    
    # Percentiles
    features['q25'] = np.percentile(recent, 25)
    features['q75'] = np.percentile(recent, 75)
    features['iqr'] = features['q75'] - features['q25']
    
    # Recent values
    features['last_1'] = recent[-1]
    features['last_2'] = recent[-2] if len(recent) > 1 else recent[-1]
    features['last_3'] = recent[-3] if len(recent) > 2 else recent[-1]
    
    # Moving averages
    features['ma_3'] = np.mean(recent[-3:])
    features['ma_5'] = np.mean(recent[-5:])
    features['ma_all'] = np.mean(recent)
    
    # Weighted moving average (recent values weighted more)
    weights = np.exp(np.linspace(-1., 0., len(recent)))
    weights /= weights.sum()
    features['wma'] = np.sum(weights * recent)
    
    # Exponential moving average
    alpha = 0.3
    ema = recent[0]
    for val in recent[1:]:
        ema = alpha * val + (1 - alpha) * ema
    features['ema'] = ema
    
    # Trend features
    x = np.arange(len(recent))
    
    # Linear trend
    if len(recent) >= 3:
        z = np.polyfit(x, recent, 1)
        features['trend_slope'] = z[0]
        features['trend_intercept'] = z[1]
        features['trend_prediction'] = z[0] * len(recent) + z[1]
    else:
        features['trend_slope'] = 0
        features['trend_intercept'] = recent[-1]
        features['trend_prediction'] = recent[-1]
    
    # Momentum
    features['momentum_1'] = recent[-1] - recent[-2] if len(recent) > 1 else 0
    features['momentum_3'] = np.mean(recent[-3:]) - np.mean(recent[-6:-3]) if len(recent) > 5 else 0
    
    # Volatility
    if len(recent) > 1:
        returns = np.diff(recent) / (recent[:-1] + 1)
        features['volatility'] = np.std(returns)
        features['mean_return'] = np.mean(returns)
    else:
        features['volatility'] = 0
        features['mean_return'] = 0
    
    # Seasonality indicators (if full lookback available)
    if len(time_series) >= 20:
        # Compare recent period to same period in previous cycles
        mid_period = time_series[-20:-10]
        features['vs_previous_period'] = np.mean(recent) / (np.mean(mid_period) + 1)
    else:
        features['vs_previous_period'] = 1
    
    # Growth rate
    if features['mean'] > 0:
        features['growth_rate'] = (recent[-1] - features['mean']) / features['mean']
    else:
        features['growth_rate'] = 0
    
    # Acceleration (change in trend)
    if len(recent) >= 6:
        recent_trend = np.mean(recent[-3:]) - np.mean(recent[-6:-3])
        features['acceleration'] = recent_trend
    else:
        features['acceleration'] = 0
    
    return features

print("   Creating advanced features for training...")

# Prepare training data using historical backtesting
X_train_list = []
y_train_list = []
metadata_list = []

lookback = 15  # Increased lookback for better patterns

for idx, row in df.iterrows():
    country = row['Country / Region']
    disease = row['Disease']
    
    time_series = np.array([row[year] for year in year_columns])
    
    # Skip if insufficient data
    if np.isnan(time_series).any() or time_series.sum() == 0:
        continue
    
    # Create sliding window training samples
    for i in range(lookback, len(time_series) - 1):
        historical = time_series[:i]
        target = time_series[i]
        
        features = create_advanced_features(historical, lookback=min(lookback, len(historical)))
        
        if features is not None and target >= 0:
            X_train_list.append(features)
            y_train_list.append(target)
            metadata_list.append({
                'country': country,
                'disease': disease,
                'year': year_columns[i]
            })

X_train_df = pd.DataFrame(X_train_list)
y_train = np.array(y_train_list)

print(f"   ✓ Created {len(X_train_df)} training samples")
print(f"   ✓ Features: {len(X_train_df.columns)}")

# Handle any remaining NaN or inf
X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan)
X_train_df = X_train_df.fillna(0)

# Feature scaling
print("\n3. Scaling features...")
scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_train_scaled = scaler.fit_transform(X_train_df)

# Train multiple advanced models
print("\n4. Training advanced ML models...")

models = {
    'XGBoost': XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    ),
    'ExtraTrees': ExtraTreesRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
}

trained_models = {}
model_scores = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                 cv=tscv, scoring='r2', n_jobs=-1)
    
    # Train on full data
    model.fit(X_train_scaled, y_train)
    
    trained_models[name] = model
    model_scores[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"      CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Select best performing models for ensemble
print("\n5. Creating optimized ensemble...")
sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['cv_mean'], reverse=True)

print("\n   Model Rankings:")
for i, (name, scores) in enumerate(sorted_models, 1):
    print(f"      {i}. {name}: R² = {scores['cv_mean']:.4f}")

# Use top 3 models with weighted ensemble
top_models = dict(sorted_models[:3])
ensemble_weights = {}
total_score = sum(s['cv_mean'] for s in top_models.values())

for name, scores in top_models.items():
    weight = scores['cv_mean'] / total_score
    ensemble_weights[name] = weight
    print(f"\n   {name} weight: {weight:.3f}")

# Make predictions for 2025
print("\n" + "=" * 80)
print("GENERATING 2025 PREDICTIONS WITH ADVANCED MODEL")
print("=" * 80)

predictions_2025 = []

for idx, row in df.iterrows():
    country = row['Country / Region']
    disease = row['Disease']
    
    time_series = np.array([row[year] for year in year_columns])
    
    if np.isnan(time_series).any() or time_series.sum() == 0:
        continue
    
    # Create features for prediction
    features = create_advanced_features(time_series, lookback=lookback)
    
    if features is None:
        continue
    
    # Convert to dataframe
    X_pred = pd.DataFrame([features])
    X_pred = X_pred.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Ensure same features as training
    for col in X_train_df.columns:
        if col not in X_pred.columns:
            X_pred[col] = 0
    
    X_pred = X_pred[X_train_df.columns]
    
    # Scale features
    X_pred_scaled = scaler.transform(X_pred)
    
    # Ensemble prediction
    ensemble_pred = 0
    individual_preds = {}
    
    for model_name, weight in ensemble_weights.items():
        model = trained_models[model_name]
        pred = model.predict(X_pred_scaled)[0]
        pred = max(0, pred)  # No negative predictions
        individual_preds[model_name] = pred
        ensemble_pred += weight * pred
    
    # Calculate prediction interval using model variance
    pred_std = np.std(list(individual_preds.values()))
    lower_bound = max(0, ensemble_pred - 1.96 * pred_std)  # 95% CI
    upper_bound = ensemble_pred + 1.96 * pred_std
    
    # Calculate outbreak risk
    recent_avg = np.mean(time_series[-3:])
    last_year = time_series[-1]
    
    if recent_avg > 0:
        increase_ratio = ensemble_pred / recent_avg
        
        if increase_ratio > 1.5 and ensemble_pred > 100:
            outbreak_risk = 'High'
        elif increase_ratio > 1.3 and ensemble_pred > 50:
            outbreak_risk = 'Medium'
        elif increase_ratio > 1.1:
            outbreak_risk = 'Low-Medium'
        else:
            outbreak_risk = 'Low'
    else:
        outbreak_risk = 'Low'
    
    # Determine confidence based on model agreement
    pred_range = max(individual_preds.values()) - min(individual_preds.values())
    if pred_range < ensemble_pred * 0.2:
        confidence = 'High'
    elif pred_range < ensemble_pred * 0.5:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    
    predictions_2025.append({
        'Country': country,
        'Disease': disease,
        'Last_Year_2024': last_year,
        'Avg_Last_3_Years': recent_avg,
        'Predicted_2025': round(ensemble_pred, 2),
        'Predicted_Lower_95CI': round(lower_bound, 2),
        'Predicted_Upper_95CI': round(upper_bound, 2),
        'Prediction_StdDev': round(pred_std, 2),
        'Trend': 'Increasing' if ensemble_pred > recent_avg else 'Decreasing',
        'Outbreak_Risk': outbreak_risk,
        'Model_Confidence': confidence,
        'XGBoost_Pred': round(individual_preds.get('XGBoost', 0), 2),
        'LightGBM_Pred': round(individual_preds.get('LightGBM', 0), 2),
        'RandomForest_Pred': round(individual_preds.get('RandomForest', 0), 2)
    })

predictions_df = pd.DataFrame(predictions_2025)

print(f"\n✓ Generated {len(predictions_df)} predictions for 2025")

# Save predictions
output_file = 'advanced_disease_predictions_2025.csv'
predictions_df.to_csv(output_file, index=False)
print(f"✓ Saved to: {output_file}")

# Analysis
print("\n" + "=" * 80)
print("PREDICTION ANALYSIS")
print("=" * 80)

print("\nOutbreak Risk Distribution:")
print(predictions_df['Outbreak_Risk'].value_counts().to_string())

print("\nModel Confidence Distribution:")
print(predictions_df['Model_Confidence'].value_counts().to_string())

# High risk outbreaks
high_risk = predictions_df[predictions_df['Outbreak_Risk'] == 'High'].sort_values('Predicted_2025', ascending=False)

print("\n" + "=" * 80)
print(f"HIGH RISK OUTBREAK PREDICTIONS ({len(high_risk)} scenarios)")
print("=" * 80)

if len(high_risk) > 0:
    print("\nTop 15 High-Risk Predictions:")
    print(high_risk[['Country', 'Disease', 'Last_Year_2024', 'Predicted_2025', 
                     'Predicted_Upper_95CI', 'Model_Confidence']].head(15).to_string(index=False))

# Feature importance
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (Top 10)")
print("=" * 80)

if 'XGBoost' in trained_models:
    xgb_model = trained_models['XGBoost']
    feature_importance = pd.DataFrame({
        'Feature': X_train_df.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + feature_importance.head(10).to_string(index=False))

# Summary by disease
print("\n" + "=" * 80)
print("PREDICTIONS BY DISEASE")
print("=" * 80)

summary = predictions_df.groupby('Disease').agg({
    'Predicted_2025': ['sum', 'mean', 'max'],
    'Country': 'count'
}).round(2)
summary.columns = ['Total_Predicted', 'Avg_Predicted', 'Max_Predicted', 'Countries']

print("\n" + summary.to_string())

# Global comparison
total_2024 = predictions_df['Last_Year_2024'].sum()
total_2025 = predictions_df['Predicted_2025'].sum()
change = ((total_2025 - total_2024) / total_2024 * 100) if total_2024 > 0 else 0

print("\n" + "=" * 80)
print("GLOBAL OUTLOOK 2025")
print("=" * 80)
print(f"Total cases 2024: {total_2024:,.0f}")
print(f"Predicted 2025: {total_2025:,.0f}")
print(f"Change: {change:+.2f}%")

print("\n" + "=" * 80)
print("✓ ADVANCED MODEL PREDICTION COMPLETE")
print("=" * 80)
print(f"\nKey Improvements:")
print(f"  • Used {len(X_train_df)} training samples (vs ~2,377 in basic model)")
print(f"  • {len(X_train_df.columns)} advanced features (vs 10 basic features)")
print(f"  • Ensemble of top 3 ML models (weighted by CV performance)")
print(f"  • 95% confidence intervals included")
print(f"  • Time series cross-validation performed")
print(f"  • Feature importance analysis")
