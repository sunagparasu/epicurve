import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DISEASE OUTBREAK PREDICTION USING MLP (NEURAL NETWORKS)")
print("Multi-Layer Perceptron for Time Series Forecasting")
print("=" * 80)

# Load data
print("\n1. Loading imputed data...")
df = pd.read_csv('combined_disease_data_imputed.csv')
print(f"   ✓ Loaded {len(df)} records")

# Get year columns
year_columns = [col for col in df.columns if str(col).isdigit()]
year_columns = sorted(year_columns, key=lambda x: int(x))

for year in year_columns:
    df[year] = pd.to_numeric(df[year], errors='coerce')

print(f"   ✓ Year range: {year_columns[0]} - {year_columns[-1]}")

# Feature engineering
print("\n2. Feature Engineering for MLP...")

def create_mlp_features(time_series, lookback=15):
    """Create features optimized for neural networks"""
    if len(time_series) < lookback:
        return None
    
    recent = time_series[-lookback:]
    features = {}
    
    # Raw recent values (for neural network to learn patterns)
    for i in range(min(10, lookback)):
        features[f'lag_{i+1}'] = recent[-(i+1)]
    
    # Statistical features
    features['mean'] = np.mean(recent)
    features['median'] = np.median(recent)
    features['std'] = np.std(recent)
    features['min'] = np.min(recent)
    features['max'] = np.max(recent)
    features['range'] = features['max'] - features['min']
    
    # Moving averages
    features['ma_3'] = np.mean(recent[-3:])
    features['ma_5'] = np.mean(recent[-5:])
    features['ma_10'] = np.mean(recent[-10:]) if len(recent) >= 10 else np.mean(recent)
    
    # Exponential moving average
    alpha = 0.3
    ema = recent[0]
    for val in recent[1:]:
        ema = alpha * val + (1 - alpha) * ema
    features['ema'] = ema
    
    # Trend
    x = np.arange(len(recent))
    if len(recent) >= 3:
        z = np.polyfit(x, recent, 1)
        features['trend_slope'] = z[0]
        features['trend_intercept'] = z[1]
    else:
        features['trend_slope'] = 0
        features['trend_intercept'] = recent[-1]
    
    # Momentum and volatility
    features['momentum'] = recent[-1] - recent[-3] if len(recent) > 2 else 0
    
    if len(recent) > 1:
        returns = np.diff(recent) / (recent[:-1] + 1)
        features['volatility'] = np.std(returns)
    else:
        features['volatility'] = 0
    
    # Growth metrics
    if features['mean'] > 0:
        features['growth_rate'] = (recent[-1] - features['mean']) / features['mean']
    else:
        features['growth_rate'] = 0
    
    return features

print("   Creating training dataset...")

X_train_list = []
y_train_list = []
metadata_list = []

lookback = 15

for idx, row in df.iterrows():
    country = row['Country / Region']
    disease = row['Disease']
    
    time_series = np.array([row[year] for year in year_columns])
    
    if np.isnan(time_series).any() or time_series.sum() == 0:
        continue
    
    # Create sliding window samples
    for i in range(lookback, len(time_series) - 1):
        historical = time_series[:i]
        target = time_series[i]
        
        features = create_mlp_features(historical, lookback=lookback)
        
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

# Handle NaN and inf
X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0)

# Feature scaling (important for neural networks)
print("\n3. Scaling features for neural network...")
scaler = MinMaxScaler()  # MLP works better with normalized data [0, 1]
X_train_scaled = scaler.fit_transform(X_train_df)
print(f"   ✓ Features scaled to range [0, 1]")

# Target scaling
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
print(f"   ✓ Targets scaled to range [0, 1]")

# Split data for validation
split_idx = int(len(X_train_scaled) * 0.8)
X_train_nn, X_val_nn = X_train_scaled[:split_idx], X_train_scaled[split_idx:]
y_train_nn, y_val_nn = y_train_scaled[:split_idx], y_train_scaled[split_idx:]

print(f"\n   Training samples: {len(X_train_nn):,}")
print(f"   Validation samples: {len(X_val_nn):,}")

# Build MLP models
print("\n" + "=" * 80)
print("BUILDING MLP MODELS")
print("=" * 80)

# Model 1: Sklearn MLPRegressor (Deep)
print("\n4. Training Deep MLP (3 hidden layers)...")

mlp_deep = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 regularization
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=30,
    random_state=42,
    verbose=False
)

print("   Architecture: 128 -> 64 -> 32 neurons")
mlp_deep.fit(X_train_scaled, y_train_scaled)
print(f"   ✓ Training complete (iterations: {mlp_deep.n_iter_})")

# Predictions
y_pred_deep = mlp_deep.predict(X_val_nn)
y_pred_deep_orig = y_scaler.inverse_transform(y_pred_deep.reshape(-1, 1)).ravel()
y_val_orig = y_scaler.inverse_transform(y_val_nn.reshape(-1, 1)).ravel()

deep_r2 = r2_score(y_val_orig, y_pred_deep_orig)
deep_mae = mean_absolute_error(y_val_orig, y_pred_deep_orig)
deep_rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_deep_orig))

print(f"   Validation R²: {deep_r2:.4f}")
print(f"   Validation MAE: {deep_mae:.2f}")
print(f"   Validation RMSE: {deep_rmse:.2f}")

# Model 2: Wide MLP (fewer layers, more neurons)
print("\n5. Training Wide MLP (2 hidden layers)...")

mlp_wide = MLPRegressor(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    alpha=0.0005,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=30,
    random_state=42,
    verbose=False
)

print("   Architecture: 256 -> 128 neurons")
mlp_wide.fit(X_train_scaled, y_train_scaled)
print(f"   ✓ Training complete (iterations: {mlp_wide.n_iter_})")

y_pred_wide = mlp_wide.predict(X_val_nn)
y_pred_wide_orig = y_scaler.inverse_transform(y_pred_wide.reshape(-1, 1)).ravel()

wide_r2 = r2_score(y_val_orig, y_pred_wide_orig)
wide_mae = mean_absolute_error(y_val_orig, y_pred_wide_orig)
wide_rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_wide_orig))

print(f"   Validation R²: {wide_r2:.4f}")
print(f"   Validation MAE: {wide_mae:.2f}")
print(f"   Validation RMSE: {wide_rmse:.2f}")

# Model 3: Very Deep MLP (5 hidden layers)
print("\n6. Training Very Deep MLP (5 hidden layers)...")

mlp_very_deep = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.002,  # Higher regularization for deep network
    learning_rate_init=0.001,
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=30,
    random_state=42,
    verbose=False
)

print("   Architecture: 256 -> 128 -> 64 -> 32 -> 16 neurons")
mlp_very_deep.fit(X_train_scaled, y_train_scaled)
print(f"   ✓ Training complete (iterations: {mlp_very_deep.n_iter_})")

y_pred_very_deep = mlp_very_deep.predict(X_val_nn)
y_pred_very_deep_orig = y_scaler.inverse_transform(y_pred_very_deep.reshape(-1, 1)).ravel()

very_deep_r2 = r2_score(y_val_orig, y_pred_very_deep_orig)
very_deep_mae = mean_absolute_error(y_val_orig, y_pred_very_deep_orig)
very_deep_rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_very_deep_orig))

print(f"   Validation R²: {very_deep_r2:.4f}")
print(f"   Validation MAE: {very_deep_mae:.2f}")
print(f"   Validation RMSE: {very_deep_rmse:.2f}")

# Model 3: Ensemble MLP (Combine all three)
print("\n7. Creating Ensemble MLP...")

# Weight models by validation performance
total_r2 = deep_r2 + wide_r2 + very_deep_r2

deep_weight = deep_r2 / total_r2
wide_weight = wide_r2 / total_r2
very_deep_weight = very_deep_r2 / total_r2

print(f"   Deep MLP weight: {deep_weight:.3f}")
print(f"   Wide MLP weight: {wide_weight:.3f}")
print(f"   Very Deep MLP weight: {very_deep_weight:.3f}")

y_pred_ensemble = (deep_weight * y_pred_deep_orig + 
                   wide_weight * y_pred_wide_orig + 
                   very_deep_weight * y_pred_very_deep_orig)

ensemble_r2 = r2_score(y_val_orig, y_pred_ensemble)
ensemble_mae = mean_absolute_error(y_val_orig, y_pred_ensemble)
ensemble_rmse = np.sqrt(mean_squared_error(y_val_orig, y_pred_ensemble))

print(f"   Ensemble R²: {ensemble_r2:.4f}")
print(f"   Ensemble MAE: {ensemble_mae:.2f}")
print(f"   Ensemble RMSE: {ensemble_rmse:.2f}")

# Compare models
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    'Model': ['Deep MLP (3 layers)', 'Wide MLP (2 layers)', 'Very Deep MLP (5 layers)', 'Ensemble MLP'],
    'R² Score': [deep_r2, wide_r2, very_deep_r2, ensemble_r2],
    'MAE': [deep_mae, wide_mae, very_deep_mae, ensemble_mae],
    'RMSE': [deep_rmse, wide_rmse, very_deep_rmse, ensemble_rmse]
})

print("\n" + comparison.to_string(index=False))

# Select best model
best_model_name = comparison.loc[comparison['R² Score'].idxmax(), 'Model']
print(f"\n✓ Best Model: {best_model_name}")

# Make 2025 predictions
print("\n" + "=" * 80)
print("GENERATING 2025 PREDICTIONS WITH MLP")
print("=" * 80)

predictions_2025 = []

for idx, row in df.iterrows():
    country = row['Country / Region']
    disease = row['Disease']
    
    time_series = np.array([row[year] for year in year_columns])
    
    if np.isnan(time_series).any() or time_series.sum() == 0:
        continue
    
    features = create_mlp_features(time_series, lookback=lookback)
    
    if features is None:
        continue
    
    # Prepare features
    X_pred = pd.DataFrame([features])
    X_pred = X_pred.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    for col in X_train_df.columns:
        if col not in X_pred.columns:
            X_pred[col] = 0
    
    X_pred = X_pred[X_train_df.columns]
    X_pred_scaled = scaler.transform(X_pred)
    
    # Get predictions from all three models
    pred_deep_scaled = mlp_deep.predict(X_pred_scaled)[0]
    pred_deep = y_scaler.inverse_transform([[pred_deep_scaled]])[0][0]
    
    pred_wide_scaled = mlp_wide.predict(X_pred_scaled)[0]
    pred_wide = y_scaler.inverse_transform([[pred_wide_scaled]])[0][0]
    
    pred_very_deep_scaled = mlp_very_deep.predict(X_pred_scaled)[0]
    pred_very_deep = y_scaler.inverse_transform([[pred_very_deep_scaled]])[0][0]
    
    # Ensemble prediction
    pred_ensemble = (deep_weight * pred_deep + 
                    wide_weight * pred_wide + 
                    very_deep_weight * pred_very_deep)
    pred_ensemble = max(0, pred_ensemble)  # No negative predictions
    
    # Calculate confidence interval
    pred_std = np.std([pred_deep, pred_wide, pred_very_deep])
    lower_bound = max(0, pred_ensemble - 1.96 * pred_std)
    upper_bound = pred_ensemble + 1.96 * pred_std
    
    # Recent statistics
    recent_avg = np.mean(time_series[-3:])
    last_year = time_series[-1]
    
    # Risk assessment
    if recent_avg > 0:
        increase_ratio = pred_ensemble / recent_avg
        
        if increase_ratio > 1.5 and pred_ensemble > 100:
            outbreak_risk = 'High'
        elif increase_ratio > 1.3 and pred_ensemble > 50:
            outbreak_risk = 'Medium'
        elif increase_ratio > 1.1:
            outbreak_risk = 'Low-Medium'
        else:
            outbreak_risk = 'Low'
    else:
        outbreak_risk = 'Low'
    
    # Confidence
    if pred_std < pred_ensemble * 0.15:
        confidence = 'High'
    elif pred_std < pred_ensemble * 0.35:
        confidence = 'Medium'
    else:
        confidence = 'Low'
    
    predictions_2025.append({
        'Country': country,
        'Disease': disease,
        'Last_Year_2024': last_year,
        'Avg_Last_3_Years': recent_avg,
        'MLP_Predicted_2025': round(pred_ensemble, 2),
        'MLP_Lower_95CI': round(lower_bound, 2),
        'MLP_Upper_95CI': round(upper_bound, 2),
        'Deep_MLP_Prediction': round(pred_deep, 2),
        'Wide_MLP_Prediction': round(pred_wide, 2),
        'VeryDeep_MLP_Prediction': round(pred_very_deep, 2),
        'Prediction_StdDev': round(pred_std, 2),
        'Trend': 'Increasing' if pred_ensemble > recent_avg else 'Decreasing',
        'Outbreak_Risk': outbreak_risk,
        'Confidence': confidence
    })

predictions_df = pd.DataFrame(predictions_2025)

print(f"\n✓ Generated {len(predictions_df)} predictions")

# Save results
output_file = 'mlp_disease_predictions_2025.csv'
predictions_df.to_csv(output_file, index=False)
print(f"✓ Saved to: {output_file}")

# Analysis
print("\n" + "=" * 80)
print("MLP PREDICTION ANALYSIS")
print("=" * 80)

print("\nOutbreak Risk Distribution:")
print(predictions_df['Outbreak_Risk'].value_counts().to_string())

print("\nConfidence Distribution:")
print(predictions_df['Confidence'].value_counts().to_string())

# High risk predictions
high_risk = predictions_df[predictions_df['Outbreak_Risk'] == 'High'].sort_values('MLP_Predicted_2025', ascending=False)

print(f"\n" + "=" * 80)
print(f"HIGH RISK PREDICTIONS ({len(high_risk)} scenarios)")
print("=" * 80)

if len(high_risk) > 0:
    print("\nTop 15:")
    print(high_risk[['Country', 'Disease', 'Last_Year_2024', 'MLP_Predicted_2025', 
                     'MLP_Upper_95CI', 'Confidence']].head(15).to_string(index=False))

# Summary by disease
print("\n" + "=" * 80)
print("PREDICTIONS BY DISEASE")
print("=" * 80)

summary = predictions_df.groupby('Disease').agg({
    'MLP_Predicted_2025': ['sum', 'mean', 'max'],
    'Country': 'count'
}).round(2)
summary.columns = ['Total', 'Average', 'Maximum', 'Countries']

print("\n" + summary.to_string())

# Global outlook
total_2024 = predictions_df['Last_Year_2024'].sum()
total_2025 = predictions_df['MLP_Predicted_2025'].sum()
change = ((total_2025 - total_2024) / total_2024 * 100) if total_2024 > 0 else 0

print("\n" + "=" * 80)
print("GLOBAL OUTLOOK 2025 (MLP)")
print("=" * 80)
print(f"Total cases 2024: {total_2024:,.0f}")
print(f"MLP Predicted 2025: {total_2025:,.0f}")
print(f"Change: {change:+.2f}%")

# Training history visualization
print("\n7. Saving training visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Error distribution
ax1 = axes[0, 0]
errors = np.abs(y_val_orig - y_pred_ensemble)
ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Absolute Error', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('MLP Prediction Error Distribution', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Model comparison scatter
ax2 = axes[0, 1]
sample_size = min(500, len(y_val_orig))
indices = np.random.choice(len(y_val_orig), sample_size, replace=False)
ax2.scatter(y_pred_deep_orig[indices], y_pred_wide_orig[indices], alpha=0.5, s=20)
ax2.set_xlabel('Deep MLP Predictions', fontsize=12)
ax2.set_ylabel('Wide MLP Predictions', fontsize=12)
ax2.set_title('Model Agreement Analysis', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Actual vs Predicted
ax3 = axes[1, 0]
sample_size = min(1000, len(y_val_orig))
indices = np.random.choice(len(y_val_orig), sample_size, replace=False)
ax3.scatter(y_val_orig[indices], y_pred_ensemble[indices], alpha=0.5, s=20)
ax3.plot([0, y_val_orig.max()], [0, y_val_orig.max()], 'r--', lw=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Cases', fontsize=12)
ax3.set_ylabel('Predicted Cases', fontsize=12)
ax3.set_title(f'MLP Ensemble Predictions (R² = {ensemble_r2:.4f})', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Model comparison
ax4 = axes[1, 1]
models = ['Deep\nMLP', 'Wide\nMLP', 'Very Deep\nMLP', 'Ensemble\nMLP']
r2_scores = [deep_r2, wide_r2, very_deep_r2, ensemble_r2]
colors = ['skyblue', 'lightgreen', 'gold']
bars = ax4.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('R² Score', fontsize=12)
ax4.set_title('MLP Model Comparison', fontsize=14, fontweight='bold')
ax4.set_ylim([0, max(r2_scores) * 1.2])
ax4.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('mlp_model_results.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved to: mlp_model_results.png")

print("\n" + "=" * 80)
print("✓ MLP PREDICTION COMPLETE")
print("=" * 80)

print(f"\nMLP Model Performance:")
print(f"  • Best R² Score: {ensemble_r2:.4f}")
print(f"  • Training Samples: {len(X_train_scaled):,}")
print(f"  • Features Used: {X_train_scaled.shape[1]}")
print(f"  • Architecture: Deep Neural Network with Batch Normalization & Dropout")
print(f"  • Ensemble of 2 MLP models for robust predictions")
