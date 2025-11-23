# Epicurve

Epidemic Pattern Analyzer — a Streamlit app to compare simulated SIR epidemic curves
with historical country-level disease incidence using engineered features and
machine-learning classifiers.

This repository contains the app entrypoint `app.py`, a `data/` directory expected to
contain Excel case/incidence files for several diseases, a Python virtual environment
folder in `epicurve/` (optional/committed here), and `requirements.txt` for dependencies.

**Quick Contents**

- `app.py` — Streamlit application with SIR simulation utilities, feature extraction,
  model training, and dashboard UI.
- `data/` — Place Excel files used by `load_all_data()` here. Filenames must match
  those referenced in `app.py` (or update `load_all_data()` to point to your files).
- `requirements.txt` — Python package dependencies.

**Features**

- Loads real-world disease incidence tables and extracts 12-element descriptors
  from short time series (per-country/year data).
- Trains a suite of ML models (Random Forest, Gradient Boosting, XGBoost, LightGBM,
  SVM, MLP, DecisionTree, GaussianNB) and builds ensemble classifiers (Voting,
  Stacking) to predict which historical disease a simulated SIR curve resembles.
- Interactive Streamlit dashboard to tune SIR parameters, visualize curves, see
  top disease predictions and similar historical country patterns.

## Prerequisites

- Linux (tested in this workspace)
- Python 3.11+ (the repo includes a virtualenv under `epicurve/` named accordingly)
- `pip` and ability to install binaries required by some ML libraries (XGBoost,
  LightGBM may require build tools; prefer installing wheels where available).

## Setup (recommended)

1. From project root (`/home/sunag/Documents/epicurve`) activate your virtualenv or create one:

```bash
# either use the included venv (if appropriate):
source epicurve/bin/activate

# or create a new venv:
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. Ensure the `data/` directory contains the Excel files expected by the app. By
default `app.load_all_data()` looks for several files (e.g. `Pertussis_...xlsx`,
`Measles_...xlsx`, etc.) — you can either copy your files into `data/` with
those names or modify `app.py` to reference different filenames.

## Running the app (Streamlit)

From the project root, after activating the environment:

```bash
# start the Streamlit app
streamlit run app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`) in your browser.

## Inspecting features and labels in a Jupyter notebook

To inspect `self.features_scaled` and `self.encoded_labels` from the app in a
notebook, follow these steps:

1. Start Jupyter Lab / Notebook from the project root:

```bash
jupyter lab
# or
jupyter notebook
```

2. In a notebook cell:

```python
import os
os.chdir('/home/sunag/Documents/epicurve')

from app import DiseaseAnalysisApp
app = DiseaseAnalysisApp()

# Load and prepare data (these functions log to Streamlit; they still work in notebook)
assert app.load_all_data()  # returns True if files were found and loaded
assert app.preprocess_data()  # returns True if features were successfully extracted

# Inspect arrays
print('features_scaled shape:', app.features_scaled.shape)
print('encoded_labels shape:', app.encoded_labels.shape)

import pandas as pd
pd.DataFrame(app.features_scaled).head()

# Show mapping for encoded labels
print('label encoder classes:', app.label_encoder.classes_)
```

## File / Directory structure

```
/home/sunag/Documents/epicurve
├── app.py
├── requirements.txt
├── README.md
├── data/                 # Excel files expected here
└── epicurve/             # included virtualenv (optional)
```

## Troubleshooting

- If `load_all_data()` cannot find any files, you'll see Streamlit errors. Check
  that `data/` exists and the files are named as expected. You can also edit
  `app.py` to change the `data_files` mapping inside `load_all_data()`.
- If installing `xgboost` or `lightgbm` fails, try installing platform wheels or
  use `pip install --no-binary :all: package` only if you have required compilers.
- If models take a long time to train, reduce `n_estimators` in the `train_models()`
  definitions or set `n_jobs` to a smaller value.

## Development notes

- `self.extract_advanced_features()` produces a fixed-length 12-element feature
  vector from a short timeseries (the ML models expect 12 features per sample).
- `self.preprocess_data()` performs country selection, numeric cleaning, feature
  extraction, label encoding and scaling; on success it populates
  `self.features`, `self.features_scaled`, `self.labels`, and `self.encoded_labels`.

## Contributing

- Open an issue or PR describing your change. Keep changes focused and include
  tests or manual verification steps where appropriate.

## License

Include your chosen license here if you want to release the project publicly.

---

If you'd like, I can:
- Add example data or a tiny synthetic dataset and update `data/` for easy demo runs.
- Add a `DEV_NOTES.md` with instructions for debugging model training and reducing
  compute time.

## Additional Scripts

This repository also contains (or may be extended with) several supplementary
scripts for experimenting with alternative models and visualizations. The
descriptions below explain their purpose and include example commands to run
them from the project root. Make sure your virtual environment is activated and
dependencies from `requirements.txt` are installed before running any script.

- `advanced_outbreak_prediction.py`
  - Description: Implements more advanced outbreak forecasting features
    (for example multi-compartment models, parameter sweeps, or probabilistic
    forecasting). Expects time-series case data (CSV or Excel) and configuration
    parameters for the forecasting run.
  - Typical inputs: `data/<disease>_cases.csv` or Excel tables, optional config
    JSON/YAML.
  - Outputs: forecast CSVs, plots (PNG/HTML), and log output.
  - Run example:

    ```bash
    source epicurve/bin/activate
    python advanced_outbreak_prediction.py --data data/Measles_cases.xlsx --out forecasts/measles_forecast.csv
    ```

- `data_analysis_visualization.py`
  - Description: Exploratory data analysis and visualization utilities. Produces
    summary tables, distribution plots, time-series visualizations and
    interactive HTML dashboards (Plotly/Altair) from the input datasets.
  - Typical inputs: one or more disease data files in `data/`.
  - Outputs: PNG/HTML visualizations saved to `outputs/`.
  - Run example:

    ```bash
    source epicurve/bin/activate
    python data_analysis_visualization.py --input data/Measles_reported_cases.xlsx --out outputs/measles_plots/
    ```

- `mlp_outbreak_prediction.py`
  - Description: Train and evaluate a multilayer perceptron (MLP) model for
    outbreak pattern recognition. Uses engineered features (the same 12-feature
    descriptor used by the main app) and outputs trained model artifacts and
    evaluation metrics.
  - Typical inputs: preprocessed feature CSV or raw data files; hyperparameter
    options can be passed via CLI arguments.
  - Outputs: trained model file (pickle), classification report, and a results CSV.
  - Run example:

    ```bash
    source epicurve/bin/activate
    python mlp_outbreak_prediction.py --features data/features.csv --labels data/labels.csv --epochs 100
    ```

- `rnn_lstm_prediction.py`
  - Description: Sequence-based forecasting using recurrent neural networks
    (LSTM). Useful when temporal order and short sequences are important for
    predicting future case counts. Typically requires longer time-series data
    than the 4–6-year country summaries used elsewhere.
  - Typical inputs: long time-series CSVs with date-indexed case counts per
    country or location.
  - Outputs: saved model checkpoints, forecast CSVs, and loss/metric plots.
  - Run example:

    ```bash
    source epicurve/bin/activate
    python rnn_lstm_prediction.py --input data/cases_timeseries.csv --epochs 50 --save_dir models/lstm/
    ```

Notes:
- The exact CLI arguments above are examples — check the top of each script or
  run `python <script>.py --help` to see supported options and required inputs.
- If any of these scripts are missing in the repository, you can use the
  examples above as templates to create them; I can generate starter versions on request.
