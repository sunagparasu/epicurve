import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from datetime import datetime
import io
import os
import warnings

warnings.filterwarnings('ignore')


class DiseaseAnalysisApp:
    def __init__(self):
        self.all_data = {}
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.disease_info = {
            'Pertussis': {'r0_range': (5, 18), 'incubation': '7-10 days',
                          'description': 'Whooping cough - bacterial respiratory infection'},
            'Mumps': {'r0_range': (4, 7), 'incubation': '16-18 days',
                      'description': 'Viral infection of salivary glands'},
            'Measles': {'r0_range': (12, 18), 'incubation': '10-12 days',
                        'description': 'Highly contagious viral infection'},
            'Rubella': {'r0_range': (5, 7), 'incubation': '14-21 days', 'description': 'Viral infection with rash'},
            'Tetanus': {'r0_range': (0, 0), 'incubation': '3-21 days',
                        'description': 'Bacterial infection through wounds'},
            'Typhoid': {'r0_range': (2, 5), 'incubation': '8-14 days', 'description': 'Bacterial intestinal infection'},
            'Polio': {'r0_range': (5, 7), 'incubation': '7-14 days', 'description': 'Viral nervous system infection'},
            'Diphtheria': {'r0_range': (6, 7), 'incubation': '2-5 days',
                           'description': 'Bacterial respiratory infection'},
            'Meningococcal': {'r0_range': (2, 4), 'incubation': '3-4 days', 'description': 'Meningococcal infection'},
            'Yellow_Fever': {'r0_range': (3, 6), 'incubation': '3-6 days', 'description': 'Viral hemorrhagic fever'}
        }
        self.setup_page()

    def setup_page(self):
        st.set_page_config(
            page_title="Epidemic Pattern Analyzer",
            page_icon="🦠",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("Epidemic Pattern Analyzer")
        st.markdown(
            "Analyze epidemic patterns using AI and SIR models. Enter epidemic parameters and get similarity analysis with real diseases.")

    def load_all_data(self):
        try:
            data_files = {
                'Pertussis': 'Pertussis_reported_cases_and_incidence_2025_18_11_13_34_UTC_1.xlsx',
                'Mumps': 'Mumps reported cases and incidence 2025-18-11 13-36 UTC.xlsx',
                'Measles': 'Measles reported cases and incidence 2025-18-11 13-23 UTC.xlsx',
                'Rubella': 'Rubella reported cases and incidence 2025-18-11 13-53 UTC.xlsx',
                'Tetanus': 'Tetanus reported cases and incidence 2025-18-11 13-22 UTC.xlsx',
                'Typhoid': 'Typhoid reported cases and incidence 2025-18-11 13-25 UTC.xlsx',
                'Polio': 'Poliomyelitis_reported_cases_and_incidence_2025_18_11_13_31_UTC.xlsx',
                'Diphtheria': 'Diphtheria_reported_cases_and_incidence_2025_18_11_14_00_UTC.xlsx',
                'Meningococcal': 'Invasive_meningococcal_disease_reported_cases_and_incidence_2025.xlsx',
                'Yellow_Fever': 'Yellow_Fever_YF_reported_cases_and_incidence_2025_18_11_13_40_UTC.xlsx'
            }

            data_dir = 'data'
            if not os.path.exists(data_dir):
                st.error(f"Data directory '{data_dir}' does not exist!")
                return False

            loaded_count = 0
            for disease, filename in data_files.items():
                filepath = os.path.join(data_dir, filename)
                if os.path.exists(filepath):
                    try:
                        self.all_data[disease] = pd.read_excel(filepath, engine='openpyxl')
                        loaded_count += 1
                    except Exception as e:
                        st.error(f"Error reading {filename}: {e}")
                else:
                    st.warning(f"File {filename} not found")

            if loaded_count == 0:
                st.error("Could not load any data files!")
                return False

            st.success(f"Successfully loaded {loaded_count} files")
            return True

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False

    def clean_numeric_value(self, value):
        if pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace(' ', '').replace(',', '').replace('<', '').replace(' ', '')
            if cleaned == '' or cleaned.lower() in ['na', 'nan', 'null']:
                return 0.0
            try:
                return float(cleaned)
            except:
                return 0.0
        return 0.0

    def extract_advanced_features(self, cases):
        """Extract advanced features from case data with better statistical properties"""
        if len(cases) == 0:
            return np.zeros(12)

        features = []
        cases = np.array(cases, dtype=float)

        try:
            # Remove zeros and very small values to improve statistical properties
            cases_clean = cases[cases > 0.1]
            if len(cases_clean) < 2:
                cases_clean = cases

            # Basic statistics (robust)
            features.extend([
                np.mean(cases_clean) if len(cases_clean) > 0 else 0.0,
                np.std(cases_clean) if len(cases_clean) > 1 else 0.0,
                np.min(cases_clean) if len(cases_clean) > 0 else 0.0,
                np.max(cases_clean) if len(cases_clean) > 0 else 0.0,
                np.median(cases_clean) if len(cases_clean) > 0 else 0.0,
            ])

            # Trend and pattern features
            if len(cases) > 2:
                x = np.arange(len(cases))

                # Linear trend
                slope, intercept = np.polyfit(x, cases, 1)
                features.append(slope)

                # Volatility and changes
                if len(cases) > 1:
                    pct_changes = np.diff(cases) / (np.abs(cases[:-1]) + 1e-8)
                    features.extend([
                        np.mean(pct_changes) if len(pct_changes) > 0 else 0.0,
                        np.std(pct_changes) if len(pct_changes) > 1 else 0.0,
                    ])
                else:
                    features.extend([0.0, 0.0])

                # Distribution shape
                features.extend([
                    (np.max(cases) - np.min(cases)) / (np.mean(cases) + 1e-8) if np.mean(cases) > 0 else 0.0,
                    # Coefficient of variation
                    len(cases_clean) / len(cases) if len(cases) > 0 else 0.0,  # Non-zero ratio
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # Ensure we have exactly 12 features
            while len(features) < 12:
                features.append(0.0)

        except Exception as e:
            # Return zeros if feature extraction fails
            return np.zeros(12)

        return np.array(features[:12], dtype=float)

    def preprocess_data(self):
        try:
            features = []
            labels = []
            countries = []
            diseases = []

            for disease_name, data in self.all_data.items():
                if len(data.columns) < 2:
                    continue

                # Find country column
                country_col = None
                for col in data.columns:
                    col_str = str(col).lower()
                    if any(keyword in col_str for keyword in ['country', 'region', 'area']):
                        country_col = col
                        break

                if not country_col:
                    country_col = data.columns[0]

                # Get recent years (focus on 2018-2023 for better data quality)
                year_columns = []
                for col in data.columns:
                    col_str = str(col)
                    if col == country_col:
                        continue
                    try:
                        if col_str.isdigit() and 2018 <= int(col_str) <= 2023:
                            year_columns.append(col)
                    except:
                        continue

                # If not enough recent years, expand range
                if len(year_columns) < 3:
                    for col in data.columns:
                        col_str = str(col)
                        if col == country_col:
                            continue
                        try:
                            if col_str.isdigit() and 2015 <= int(col_str) <= 2024:
                                year_columns.append(col)
                        except:
                            continue

                # Sort by year (newest first) and take 4-6 years
                try:
                    year_columns = sorted(year_columns, key=lambda x: int(x), reverse=True)[:6]
                except:
                    year_columns = sorted(year_columns, reverse=True)[:6]

                if len(year_columns) < 2:
                    continue

                processed_count = 0
                for idx, row in data.iterrows():
                    if processed_count >= 100:  # Increased limit
                        break

                    country = str(row[country_col])
                    # Skip regional and global data
                    if any(keyword in country.lower() for keyword in
                           ['global', 'region', 'african', 'eastern', 'european', 'americas', 'asia', 'pacific',
                            'world']):
                        continue

                    cases = []
                    valid_data_points = 0

                    for year_col in year_columns:
                        case_val = row[year_col]
                        cleaned_val = self.clean_numeric_value(case_val)
                        cases.append(cleaned_val)
                        if cleaned_val > 0:
                            valid_data_points += 1

                    # Require at least 3 valid data points and some variability
                    if valid_data_points < 3 or np.std(cases) < 1e-6:
                        continue

                    # Extract advanced features
                    case_features = self.extract_advanced_features(cases)

                    # Validate features
                    if np.any(np.isnan(case_features)) or np.any(np.isinf(case_features)):
                        continue

                    features.append(case_features)
                    labels.append(disease_name)
                    countries.append(country)
                    diseases.append(disease_name)
                    processed_count += 1

            if len(features) == 0:
                st.error("Could not extract features from data")
                return False

            # Convert to numpy array
            self.features = np.array(features, dtype=float)
            self.labels = labels
            self.countries = countries
            self.diseases = diseases

            # Remove any invalid values
            valid_indices = ~np.any(np.isnan(self.features) | np.isinf(self.features), axis=1)
            self.features = self.features[valid_indices]
            self.labels = [self.labels[i] for i in range(len(valid_indices)) if valid_indices[i]]
            self.countries = [self.countries[i] for i in range(len(valid_indices)) if valid_indices[i]]
            self.diseases = [self.diseases[i] for i in range(len(valid_indices)) if valid_indices[i]]

            if len(self.features) == 0:
                st.error("No valid features after cleaning")
                return False

            # Encode labels and scale features
            self.encoded_labels = self.label_encoder.fit_transform(self.labels)
            self.features_scaled = self.scaler.fit_transform(self.features)

            st.success(f"Data prepared: {len(self.features)} samples, {len(np.unique(self.labels))} diseases")

            # Show data distribution
            st.subheader("Data Distribution")
            dist_df = pd.DataFrame({'Disease': self.labels})
            st.dataframe(dist_df['Disease'].value_counts(), width='stretch')

            return True

        except Exception as e:
            st.error(f"Error preprocessing data: {e}")
            return False

    def train_models(self):
        try:
            st.info("Training optimized models...")

            if len(np.unique(self.encoded_labels)) < 2:
                st.error("Not enough classes for training. Need at least 2 different diseases.")
                return False

            X_train, X_test, y_train, y_test = train_test_split(
                self.features_scaled, self.encoded_labels, test_size=0.2, random_state=42, stratify=self.encoded_labels
            )

            # Define optimized models
            base_models = [
                ('rf', RandomForestClassifier(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    random_state=42
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=8,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    subsample=0.8,
                    random_state=42
                )),
                ('xgb', XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=8,
                    min_child_weight=1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='mlogloss'
                )),
                ('lgb', LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=8,
                    num_leaves=31,
                    min_child_samples=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )),
                ('svm', SVC(
                    probability=True,
                    kernel='rbf',
                    C=2.0,
                    gamma='scale',
                    random_state=42
                )),
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=(256, 128, 64),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    learning_rate='adaptive',
                    max_iter=3000,
                    early_stopping=True,
                    random_state=42
                )),
                ('dt', DecisionTreeClassifier(
                    max_depth=20,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    random_state=42
                )),
                ('nb', GaussianNB())
            ]

            # Train individual models
            for name, model in base_models:
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)

                    # Cross-validation with fewer folds for stability
                    cv_scores = cross_val_score(model, self.features_scaled, self.encoded_labels,
                                                cv=3, scoring='accuracy', n_jobs=-1)

                    self.models[name] = {
                        'model': model,
                        'score': score,
                        'cv_mean': float(np.mean(cv_scores)),
                        'cv_std': float(np.std(cv_scores))
                    }
                    st.success(
                        f"✅ {name.upper()} trained - Accuracy: {score:.3f}, CV: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

                except Exception as e:
                    st.error(f"❌ Error training {name}: {e}")

            # Create ensemble models with proper cross-validation
            try:
                # Use models with reasonable performance
                best_models = [(name, model['model']) for name, model in self.models.items()
                               if model['score'] > 0.15]

                if len(best_models) >= 2:
                    # Voting Classifier
                    voting_clf = VotingClassifier(estimators=best_models, voting='soft')
                    voting_clf.fit(X_train, y_train)
                    voting_score = accuracy_score(y_test, voting_clf.predict(X_test))
                    cv_scores_voting = cross_val_score(voting_clf, self.features_scaled, self.encoded_labels,
                                                       cv=3, scoring='accuracy', n_jobs=-1)
                    self.models['voting'] = {
                        'model': voting_clf,
                        'score': voting_score,
                        'cv_mean': float(np.mean(cv_scores_voting)),
                        'cv_std': float(np.std(cv_scores_voting))
                    }
                    st.success(f"✅ VOTING ensemble trained - Accuracy: {voting_score:.3f}")

                    # Stacking Classifier
                    stacking_clf = StackingClassifier(
                        estimators=best_models,
                        final_estimator=LogisticRegression(max_iter=2000, random_state=42),
                        cv=3,
                        n_jobs=-1
                    )
                    stacking_clf.fit(X_train, y_train)
                    stacking_score = accuracy_score(y_test, stacking_clf.predict(X_test))
                    cv_scores_stacking = cross_val_score(stacking_clf, self.features_scaled, self.encoded_labels,
                                                         cv=3, scoring='accuracy', n_jobs=-1)
                    self.models['stacking'] = {
                        'model': stacking_clf,
                        'score': stacking_score,
                        'cv_mean': float(np.mean(cv_scores_stacking)),
                        'cv_std': float(np.std(cv_scores_stacking))
                    }
                    st.success(f"✅ STACKING ensemble trained - Accuracy: {stacking_score:.3f}")

            except Exception as e:
                st.warning(f"Could not create ensemble models: {e}")

            # Show results with proper data types
            if self.models:
                st.subheader("Model Performance Summary")
                results = []
                for model_name, info in self.models.items():
                    results.append({
                        'Model': model_name.upper(),
                        'Accuracy': float(info['score']),
                        'CV_Mean': float(info.get('cv_mean', 0.0)),
                        'CV_Std': float(info.get('cv_std', 0.0))
                    })

                results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

                # Display with proper formatting
                st.dataframe(
                    results_df.style.format({
                        'Accuracy': '{:.3f}',
                        'CV_Mean': '{:.3f}',
                        'CV_Std': '{:.3f}'
                    }),
                    width='stretch'
                )

                # Show best model
                best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['score'])
                best_score = self.models[best_model_name]['score']
                st.success(f"🎯 Best model: {best_model_name.upper()} with accuracy: {best_score:.3f}")

                return True
            else:
                st.error("No models were successfully trained")
                return False

        except Exception as e:
            st.error(f"Error training models: {e}")
            return False

    def sir_model(self, y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    def generate_sir_curve(self, N, S0, I0, R0, beta, gamma, days=365):
        y0 = S0, I0, R0
        t = np.linspace(0, days, days)
        result = odeint(self.sir_model, y0, t, args=(N, beta, gamma))
        S, I, R = result.T
        return t, S, I, R

    def calculate_r0(self, beta, gamma):
        if gamma == 0:
            return float('inf')
        return beta / gamma

    def analyze_pattern(self, beta, gamma, S0, I0, R0, N, selected_model='rf'):
        if not self.models:
            st.error("Models not trained!")
            return None, None, None, None, pd.DataFrame(), np.array([])

        # Generate SIR curve
        t, S, I, R = self.generate_sir_curve(N, S0, I0, R0, beta, gamma, days=365)

        # Sample multiple points from the curve
        sample_points = np.linspace(0, len(I) - 1, min(8, len(I)), dtype=int)
        sir_cases = I[sample_points]

        # Extract features from SIR curve
        sir_features = self.extract_advanced_features(sir_cases)

        try:
            sir_features_scaled = self.scaler.transform([sir_features])
        except Exception as e:
            st.error(f"Normalization error: {e}")
            return t, S, I, R, pd.DataFrame(), sir_cases

        model_info = self.models[selected_model]
        model = model_info['model']

        # Get probabilities
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(sir_features_scaled)[0]
            else:
                decision = model.decision_function(sir_features_scaled)[0]
                probabilities = np.exp(decision) / np.sum(np.exp(decision))
        except Exception as e:
            n_classes = len(self.label_encoder.classes_)
            probabilities = np.ones(n_classes) / n_classes

        # Create results
        results = []
        calculated_r0 = self.calculate_r0(beta, gamma)

        for disease, prob in zip(self.label_encoder.classes_, probabilities):
            disease_data = self.disease_info.get(disease, {})
            r0_range = disease_data.get('r0_range', (0, 0))

            # Calculate R0 match score
            r0_match = 0
            if r0_range[1] > 0:
                if calculated_r0 >= r0_range[0] and calculated_r0 <= r0_range[1]:
                    r0_match = 100
                elif calculated_r0 < r0_range[0]:
                    r0_match = max(0, (calculated_r0 / r0_range[0]) * 50)
                else:
                    r0_match = max(0, (r0_range[1] / calculated_r0) * 50)

            # Combined score (weighted average)
            combined_score = (prob * 100 * 0.7) + (r0_match * 0.3)

            results.append({
                'Disease': disease,
                'Probability': prob * 100,
                'R0_Match': r0_match,
                'Combined_Score': combined_score,
                'R0_Range': f"{r0_range[0]}-{r0_range[1]}",
                'Description': disease_data.get('description', 'No description')
            })

        results_df = pd.DataFrame(results).sort_values('Combined_Score', ascending=False)
        return t, S, I, R, results_df, sir_cases

    def find_similar_countries(self, sir_cases, top_n=5):
        similarities = []

        if not hasattr(self, 'features_scaled') or len(self.features_scaled) == 0:
            return pd.DataFrame(columns=['Country', 'Disease', 'Similarity'])

        sir_features = self.extract_advanced_features(sir_cases)
        if len(sir_features) == 0:
            return pd.DataFrame(columns=['Country', 'Disease', 'Similarity'])

        try:
            sir_features_scaled = self.scaler.transform([sir_features])[0]
        except:
            return pd.DataFrame(columns=['Country', 'Disease', 'Similarity'])

        for i, (country, disease, real_features) in enumerate(zip(self.countries, self.diseases, self.features_scaled)):
            try:
                similarity = np.dot(sir_features_scaled, real_features) / (
                        np.linalg.norm(sir_features_scaled) * np.linalg.norm(real_features)
                )
                similarity_percent = max(0, min(100, similarity * 100))

                similarities.append({
                    'Country': country,
                    'Disease': disease,
                    'Similarity': similarity_percent
                })
            except:
                continue

        if similarities:
            similarities_df = pd.DataFrame(similarities).sort_values('Similarity', ascending=False).head(top_n)
            return similarities_df
        else:
            return pd.DataFrame(columns=['Country', 'Disease', 'Similarity'])

    def create_dashboard(self):
        st.sidebar.header("Epidemic Parameters")

        # Population parameters
        st.sidebar.subheader("Population Parameters")
        N = st.sidebar.number_input("Total Population (N)", min_value=1000, max_value=10000000, value=1000000,
                                    step=1000)
        S0 = st.sidebar.number_input("Initial Susceptible (S0)", min_value=0, max_value=N, value=N - 100, step=100)
        I0 = st.sidebar.number_input("Initial Infected (I0)", min_value=1, max_value=N, value=100, step=10)
        R0_val = st.sidebar.number_input("Initial Recovered (R0)", min_value=0, max_value=N, value=0, step=10)

        # Model parameters
        st.sidebar.subheader("Model Parameters")
        beta = st.sidebar.slider(
            "β (Beta) - Infection rate",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Rate of infection spread"
        )

        gamma = st.sidebar.slider(
            "γ (Gamma) - Recovery rate",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Rate of recovery/death"
        )

        # Model selection
        st.sidebar.subheader("Model Selection")
        if self.models:
            model_options = list(self.models.keys())
            # Prefer ensemble models
            if 'stacking' in model_options:
                default_index = model_options.index('stacking')
            elif 'voting' in model_options:
                default_index = model_options.index('voting')
            else:
                best_model = max(self.models.items(), key=lambda x: x[1]['score'])[0]
                default_index = model_options.index(best_model)

            selected_model = st.sidebar.selectbox(
                "Select ML Model",
                model_options,
                index=default_index,
                format_func=lambda x: f"{x.upper()} (Acc: {self.models[x]['score']:.3f})"
            )
        else:
            selected_model = 'rf'

        # Main dashboard
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("SIR Model Simulation")

            r0 = self.calculate_r0(beta, gamma)

            # Metrics
            col_r0, col_peak, col_total = st.columns(3)
            with col_r0:
                st.metric("Basic Reproduction Number (R₀)", f"{r0:.2f}")
            with col_peak:
                st.metric("Infection Rate (β)", f"{beta:.2f}")
            with col_total:
                st.metric("Recovery Rate (γ)", f"{gamma:.2f}")

            # Generate analysis
            t, S, I, R, results_df, sir_cases = self.analyze_pattern(beta, gamma, S0, I0, R0_val, N, selected_model)

            if results_df is not None and not results_df.empty:
                # SIR Plot
                fig_sir = go.Figure()
                fig_sir.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Susceptible (S)', line=dict(color='blue')))
                fig_sir.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infected (I)', line=dict(color='red')))
                fig_sir.add_trace(go.Scatter(x=t, y=R, mode='lines', name='Recovered (R)', line=dict(color='green')))

                fig_sir.update_layout(
                    title=f"SIR Model (β={beta}, γ={gamma}, R₀={r0:.2f})",
                    xaxis_title='Days',
                    yaxis_title='Number of People',
                    height=400
                )

                st.plotly_chart(fig_sir, use_container_width=True)

        with col2:
            st.subheader("Disease Prediction")

            if results_df is not None and not results_df.empty:
                # Top predictions
                top_results = results_df.head(5)

                fig_prob = px.bar(
                    top_results,
                    x='Combined_Score',
                    y='Disease',
                    orientation='h',
                    color='Combined_Score',
                    color_continuous_scale='viridis',
                    labels={'Combined_Score': 'Confidence Score (%)', 'Disease': 'Disease'}
                )

                fig_prob.update_layout(
                    title="Top 5 Disease Predictions",
                    xaxis_title='Confidence Score (%)',
                    yaxis_title='Disease',
                    height=300
                )

                st.plotly_chart(fig_prob, use_container_width=True)

                # Best prediction
                best_prediction = results_df.iloc[0]
                confidence_level = "High" if best_prediction['Combined_Score'] > 70 else "Medium" if best_prediction[
                                                                                                         'Combined_Score'] > 40 else "Low"

                st.success(f"**Prediction:** {best_prediction['Disease']} ({confidence_level} confidence)")
                st.info(f"**AI Probability:** {best_prediction['Probability']:.1f}%")
                st.info(f"**R₀ Match:** {best_prediction['R0_Match']:.1f}%")

        # Similar patterns
        if results_df is not None and not results_df.empty:
            st.subheader("Similar Historical Patterns")
            similarities_df = self.find_similar_countries(sir_cases)

            if not similarities_df.empty:
                st.dataframe(
                    similarities_df[['Country', 'Disease', 'Similarity']].style.format({'Similarity': '{:.1f}%'}),
                    width='stretch'
                )

        # Detailed analysis
        if results_df is not None and not results_df.empty:
            st.subheader("Detailed Analysis")

            # Model info
            if selected_model in self.models:
                model_info = self.models[selected_model]
                st.write(f"**Selected Model:** {selected_model.upper()}")
                st.write(f"**Model Accuracy:** {model_info['score']:.3f}")
                if 'cv_mean' in model_info:
                    st.write(f"**Cross-Validation Score:** {model_info['cv_mean']:.3f} ± {model_info['cv_std']:.3f}")

            # Epidemic metrics
            col5, col6, col7 = st.columns(3)
            with col5:
                peak_infected = np.max(I) if I is not None else 0
                st.metric("Peak Infected", f"{peak_infected:,.0f}")
            with col6:
                peak_day = np.argmax(I) if I is not None else 0
                st.metric("Peak Day", f"{peak_day}")
            with col7:
                total_cases = np.max(R) if R is not None else 0
                st.metric("Total Cases", f"{total_cases:,.0f}")

            # Complete results
            st.subheader("All Predictions")
            display_results = results_df[['Disease', 'Probability', 'R0_Match', 'Combined_Score', 'R0_Range']].copy()
            display_results.columns = ['Disease', 'AI Probability (%)', 'R₀ Match (%)', 'Confidence Score (%)',
                                       'Typical R₀ Range']
            st.dataframe(
                display_results.style.format({
                    'AI Probability (%)': '{:.1f}%',
                    'R₀ Match (%)': '{:.1f}%',
                    'Confidence Score (%)': '{:.1f}%'
                }),
                width='stretch'
            )

        # Export
        st.subheader("Export Results")
        try:
            if results_df is not None and not results_df.empty:
                csv_data = io.BytesIO()
                results_df.to_csv(csv_data, index=False)
                csv_data.seek(0)

                st.download_button(
                    label="Download Predictions (CSV)",
                    data=csv_data.getvalue(),
                    file_name=f"epidemic_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error creating export file: {e}")

    def run(self):
        if self.load_all_data():
            if self.preprocess_data():
                if self.train_models():
                    self.create_dashboard()
                else:
                    st.error("Failed to train models")
            else:
                st.error("Failed to prepare data for training")
        else:
            st.error("Failed to load data")


if __name__ == "__main__":
    app = DiseaseAnalysisApp()
    app.run()