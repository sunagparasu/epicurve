import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DISEASE OUTBREAK PREDICTION USING RNN & LSTM (PyTorch)")
print("Recurrent Neural Networks for Time Series Forecasting")
print("=" * 80)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

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

# Prepare sequences for RNN/LSTM
print("\n2. Preparing sequential data for RNN/LSTM...")

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series sequences"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def create_sequences(time_series, seq_length=10):
    """Create sequences for RNN/LSTM training"""
    sequences = []
    targets = []
    
    for i in range(len(time_series) - seq_length):
        seq = time_series[i:i+seq_length]
        target = time_series[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Prepare training data
seq_length = 10  # Use last 10 years to predict next year
all_sequences = []
all_targets = []
metadata = []

for idx, row in df.iterrows():
    country = row['Country / Region']
    disease = row['Disease']
    
    time_series = np.array([row[year] for year in year_columns])
    
    if np.isnan(time_series).any() or time_series.sum() == 0:
        continue
    
    # Create sequences
    sequences, targets = create_sequences(time_series, seq_length)
    
    if len(sequences) > 0:
        all_sequences.extend(sequences)
        all_targets.extend(targets)
        
        for i in range(len(sequences)):
            metadata.append({
                'country': country,
                'disease': disease
            })

all_sequences = np.array(all_sequences)
all_targets = np.array(all_targets)

print(f"   ✓ Created {len(all_sequences)} sequences")
print(f"   ✓ Sequence length: {seq_length} years")

# Scale data
print("\n3. Scaling data...")
scaler = MinMaxScaler()

# Reshape for scaling
all_sequences_reshaped = all_sequences.reshape(-1, 1)
all_sequences_scaled = scaler.fit_transform(all_sequences_reshaped).reshape(all_sequences.shape)

all_targets_scaled = scaler.transform(all_targets.reshape(-1, 1)).ravel()

print(f"   ✓ Data scaled to range [0, 1]")

# Reshape for RNN/LSTM: (batch, seq_length, input_size)
all_sequences_scaled = all_sequences_scaled.reshape(-1, seq_length, 1)

# Split into train/validation
split_idx = int(len(all_sequences_scaled) * 0.8)
X_train = all_sequences_scaled[:split_idx]
y_train = all_targets_scaled[:split_idx]
X_val = all_sequences_scaled[split_idx:]
y_val = all_targets_scaled[split_idx:]

print(f"\n   Training samples: {len(X_train):,}")
print(f"   Validation samples: {len(X_val):,}")

# Create datasets and dataloaders
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define RNN Model
print("\n" + "=" * 80)
print("MODEL 1: SIMPLE RNN")
print("=" * 80)

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_length, input_size)
        out, _ = self.rnn(x)
        
        # Take output from last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out.squeeze()

# Define LSTM Model
print("\n" + "=" * 80)
print("MODEL 2: LSTM (Long Short-Term Memory)")
print("=" * 80)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_length, input_size)
        out, _ = self.lstm(x)
        
        # Take output from last time step
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out.squeeze()

# Define Bidirectional LSTM
print("\n" + "=" * 80)
print("MODEL 3: BIDIRECTIONAL LSTM")
print("=" * 80)

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # *2 because bidirectional
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze()

# Training function
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=15):
    """Train the model with early stopping"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# Evaluation function
def evaluate_model(model, val_loader, scaler):
    """Evaluate model and return metrics"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in val_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
    actuals_orig = scaler.inverse_transform(actuals.reshape(-1, 1)).ravel()
    
    # Calculate metrics
    mae = mean_absolute_error(actuals_orig, predictions_orig)
    rmse = np.sqrt(mean_squared_error(actuals_orig, predictions_orig))
    r2 = r2_score(actuals_orig, predictions_orig)
    
    return mae, rmse, r2, predictions_orig, actuals_orig

# Train all models
print("\n4. Training Models...")

models_dict = {}
results_dict = {}

# Train RNN
print("\n" + "-" * 80)
print("Training Simple RNN...")
print("-" * 80)
rnn_model = SimpleRNN(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
print(f"   Parameters: {sum(p.numel() for p in rnn_model.parameters()):,}")
rnn_model, rnn_train_loss, rnn_val_loss = train_model(rnn_model, train_loader, val_loader, epochs=100)
rnn_mae, rnn_rmse, rnn_r2, rnn_preds, rnn_actuals = evaluate_model(rnn_model, val_loader, scaler)
models_dict['RNN'] = rnn_model
results_dict['RNN'] = {'mae': rnn_mae, 'rmse': rnn_rmse, 'r2': rnn_r2, 'preds': rnn_preds}
print(f"\n   RNN Results: MAE={rnn_mae:.2f}, RMSE={rnn_rmse:.2f}, R²={rnn_r2:.4f}")

# Train LSTM
print("\n" + "-" * 80)
print("Training LSTM...")
print("-" * 80)
lstm_model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
print(f"   Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
lstm_model, lstm_train_loss, lstm_val_loss = train_model(lstm_model, train_loader, val_loader, epochs=100)
lstm_mae, lstm_rmse, lstm_r2, lstm_preds, lstm_actuals = evaluate_model(lstm_model, val_loader, scaler)
models_dict['LSTM'] = lstm_model
results_dict['LSTM'] = {'mae': lstm_mae, 'rmse': lstm_rmse, 'r2': lstm_r2, 'preds': lstm_preds}
print(f"\n   LSTM Results: MAE={lstm_mae:.2f}, RMSE={lstm_rmse:.2f}, R²={lstm_r2:.4f}")

# Train Bidirectional LSTM
print("\n" + "-" * 80)
print("Training Bidirectional LSTM...")
print("-" * 80)
bilstm_model = BiLSTMModel(input_size=1, hidden_size=64, num_layers=2, dropout=0.2).to(device)
print(f"   Parameters: {sum(p.numel() for p in bilstm_model.parameters()):,}")
bilstm_model, bilstm_train_loss, bilstm_val_loss = train_model(bilstm_model, train_loader, val_loader, epochs=100)
bilstm_mae, bilstm_rmse, bilstm_r2, bilstm_preds, bilstm_actuals = evaluate_model(bilstm_model, val_loader, scaler)
models_dict['BiLSTM'] = bilstm_model
results_dict['BiLSTM'] = {'mae': bilstm_mae, 'rmse': bilstm_rmse, 'r2': bilstm_r2, 'preds': bilstm_preds}
print(f"\n   BiLSTM Results: MAE={bilstm_mae:.2f}, RMSE={bilstm_rmse:.2f}, R²={bilstm_r2:.4f}")

# Model comparison
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

comparison = pd.DataFrame({
    'Model': ['Simple RNN', 'LSTM', 'Bidirectional LSTM'],
    'MAE': [rnn_mae, lstm_mae, bilstm_mae],
    'RMSE': [rnn_rmse, lstm_rmse, bilstm_rmse],
    'R² Score': [rnn_r2, lstm_r2, bilstm_r2]
})

print("\n" + comparison.to_string(index=False))

# Select best model
best_idx = comparison['R² Score'].idxmax()
best_model_name = comparison.loc[best_idx, 'Model']
print(f"\n✓ Best Model: {best_model_name}")

# Create ensemble
print("\n5. Creating Ensemble Model...")
total_r2 = rnn_r2 + lstm_r2 + bilstm_r2
rnn_weight = rnn_r2 / total_r2
lstm_weight = lstm_r2 / total_r2
bilstm_weight = bilstm_r2 / total_r2

print(f"   RNN weight: {rnn_weight:.3f}")
print(f"   LSTM weight: {lstm_weight:.3f}")
print(f"   BiLSTM weight: {bilstm_weight:.3f}")

ensemble_preds = (rnn_weight * rnn_preds + 
                  lstm_weight * lstm_preds + 
                  bilstm_weight * bilstm_preds)

ensemble_r2 = r2_score(rnn_actuals, ensemble_preds)
ensemble_mae = mean_absolute_error(rnn_actuals, ensemble_preds)
ensemble_rmse = np.sqrt(mean_squared_error(rnn_actuals, ensemble_preds))

print(f"\n   Ensemble Results: MAE={ensemble_mae:.2f}, RMSE={ensemble_rmse:.2f}, R²={ensemble_r2:.4f}")

# Make 2025 predictions
print("\n" + "=" * 80)
print("GENERATING 2025 PREDICTIONS")
print("=" * 80)

predictions_2025 = []

for idx, row in df.iterrows():
    country = row['Country / Region']
    disease = row['Disease']
    
    time_series = np.array([row[year] for year in year_columns])
    
    if np.isnan(time_series).any() or time_series.sum() == 0:
        continue
    
    if len(time_series) < seq_length:
        continue
    
    # Get last sequence
    last_sequence = time_series[-seq_length:]
    
    # Scale
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1)).ravel()
    last_sequence_scaled = last_sequence_scaled.reshape(1, seq_length, 1)
    
    # Convert to tensor
    last_sequence_tensor = torch.FloatTensor(last_sequence_scaled).to(device)
    
    # Get predictions from all models
    with torch.no_grad():
        rnn_pred_scaled = rnn_model(last_sequence_tensor).cpu().item()
        lstm_pred_scaled = lstm_model(last_sequence_tensor).cpu().item()
        bilstm_pred_scaled = bilstm_model(last_sequence_tensor).cpu().item()
    
    # Inverse transform
    rnn_pred = scaler.inverse_transform([[rnn_pred_scaled]])[0][0]
    lstm_pred = scaler.inverse_transform([[lstm_pred_scaled]])[0][0]
    bilstm_pred = scaler.inverse_transform([[bilstm_pred_scaled]])[0][0]
    
    # Ensemble prediction
    ensemble_pred = (rnn_weight * rnn_pred + 
                     lstm_weight * lstm_pred + 
                     bilstm_weight * bilstm_pred)
    ensemble_pred = max(0, ensemble_pred)
    
    # Confidence interval
    pred_std = np.std([rnn_pred, lstm_pred, bilstm_pred])
    lower_bound = max(0, ensemble_pred - 1.96 * pred_std)
    upper_bound = ensemble_pred + 1.96 * pred_std
    
    # Statistics
    recent_avg = np.mean(time_series[-3:])
    last_year = time_series[-1]
    
    # Risk assessment
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
    
    predictions_2025.append({
        'Country': country,
        'Disease': disease,
        'Last_Year_2024': last_year,
        'Avg_Last_3_Years': recent_avg,
        'RNN_LSTM_Predicted_2025': round(ensemble_pred, 2),
        'Predicted_Lower_95CI': round(lower_bound, 2),
        'Predicted_Upper_95CI': round(upper_bound, 2),
        'RNN_Prediction': round(rnn_pred, 2),
        'LSTM_Prediction': round(lstm_pred, 2),
        'BiLSTM_Prediction': round(bilstm_pred, 2),
        'Trend': 'Increasing' if ensemble_pred > recent_avg else 'Decreasing',
        'Outbreak_Risk': outbreak_risk
    })

predictions_df = pd.DataFrame(predictions_2025)

print(f"\n✓ Generated {len(predictions_df)} predictions")

# Save predictions
output_file = 'rnn_lstm_predictions_2025.csv'
predictions_df.to_csv(output_file, index=False)
print(f"✓ Saved to: {output_file}")

# Analysis
print("\n" + "=" * 80)
print("PREDICTION ANALYSIS")
print("=" * 80)

print("\nOutbreak Risk Distribution:")
print(predictions_df['Outbreak_Risk'].value_counts().to_string())

high_risk = predictions_df[predictions_df['Outbreak_Risk'] == 'High'].sort_values('RNN_LSTM_Predicted_2025', ascending=False)

if len(high_risk) > 0:
    print(f"\nHigh Risk Predictions (Top 15):")
    print(high_risk[['Country', 'Disease', 'Last_Year_2024', 'RNN_LSTM_Predicted_2025']].head(15).to_string(index=False))

# Summary
summary = predictions_df.groupby('Disease').agg({
    'RNN_LSTM_Predicted_2025': ['sum', 'mean', 'max'],
    'Country': 'count'
}).round(2)
summary.columns = ['Total', 'Average', 'Maximum', 'Countries']

print("\n" + "=" * 80)
print("PREDICTIONS BY DISEASE")
print("=" * 80)
print("\n" + summary.to_string())

# Global outlook
total_2024 = predictions_df['Last_Year_2024'].sum()
total_2025 = predictions_df['RNN_LSTM_Predicted_2025'].sum()
change = ((total_2025 - total_2024) / total_2024 * 100) if total_2024 > 0 else 0

print("\n" + "=" * 80)
print("GLOBAL OUTLOOK 2025")
print("=" * 80)
print(f"Total cases 2024: {total_2024:,.0f}")
print(f"Predicted 2025: {total_2025:,.0f}")
print(f"Change: {change:+.2f}%")

# Visualizations
print("\n6. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Training history comparison
ax1 = axes[0, 0]
ax1.plot(rnn_val_loss, label='RNN', linewidth=2)
ax1.plot(lstm_val_loss, label='LSTM', linewidth=2)
ax1.plot(bilstm_val_loss, label='BiLSTM', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Validation Loss', fontsize=12)
ax1.set_title('Training Convergence Comparison', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Model performance comparison
ax2 = axes[0, 1]
models = ['RNN', 'LSTM', 'BiLSTM', 'Ensemble']
r2_scores = [rnn_r2, lstm_r2, bilstm_r2, ensemble_r2]
colors = ['skyblue', 'lightgreen', 'gold', 'coral']
bars = ax2.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('R² Score', fontsize=12)
ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax2.set_ylim([0, max(r2_scores) * 1.2])
ax2.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Actual vs Predicted (Ensemble)
ax3 = axes[1, 0]
sample_size = min(1000, len(ensemble_preds))
indices = np.random.choice(len(ensemble_preds), sample_size, replace=False)
ax3.scatter(rnn_actuals[indices], ensemble_preds[indices], alpha=0.5, s=20)
ax3.plot([0, rnn_actuals.max()], [0, rnn_actuals.max()], 'r--', lw=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Cases', fontsize=12)
ax3.set_ylabel('Predicted Cases', fontsize=12)
ax3.set_title(f'Ensemble Predictions (R² = {ensemble_r2:.4f})', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Error comparison
ax4 = axes[1, 1]
mae_values = [rnn_mae, lstm_mae, bilstm_mae, ensemble_mae]
rmse_values = [rnn_rmse, lstm_rmse, bilstm_rmse, ensemble_rmse]

x = np.arange(len(models))
width = 0.35

bars1 = ax4.bar(x - width/2, mae_values, width, label='MAE', alpha=0.7)
bars2 = ax4.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.7)

ax4.set_xlabel('Model', fontsize=12)
ax4.set_ylabel('Error', fontsize=12)
ax4.set_title('MAE and RMSE Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rnn_lstm_results.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved visualizations to: rnn_lstm_results.png")

print("\n" + "=" * 80)
print("✓ RNN/LSTM PREDICTION COMPLETE")
print("=" * 80)

print(f"\nBest Model Performance:")
print(f"  • Model: {best_model_name}")
print(f"  • R² Score: {comparison.loc[best_idx, 'R² Score']:.4f}")
print(f"  • MAE: {comparison.loc[best_idx, 'MAE']:.2f}")
print(f"  • RMSE: {comparison.loc[best_idx, 'RMSE']:.2f}")
print(f"\nEnsemble Performance:")
print(f"  • R² Score: {ensemble_r2:.4f}")
print(f"  • Improvement: {((ensemble_r2 - max(rnn_r2, lstm_r2, bilstm_r2)) / max(rnn_r2, lstm_r2, bilstm_r2) * 100):+.2f}%")
