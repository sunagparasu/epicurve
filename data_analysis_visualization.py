import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import warnings
import os

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class DiseaseDataVisualizer:
    def __init__(self):
        self.combined_df = None
        self.raw_data = None
        self.features_scaled = None
        self.pca_results = None
        self.setup_page()

    def setup_page(self):
        st.set_page_config(
            page_title="Disease Data Analysis & Visualization",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🦠 Disease Outbreak Data Analysis & Visualization")
        st.markdown("### Comprehensive Analysis of Combined Disease Data (1980-2024)")

    def load_all_data(self):
        """Load the combined disease dataset"""
        combined_file = 'combined_disease_data.csv'
        
        with st.spinner('Loading combined disease data...'):
            try:
                if os.path.exists(combined_file):
                    self.raw_data = pd.read_csv(combined_file)
                    st.success(f"✅ Successfully loaded combined dataset with {len(self.raw_data)} records")
                    st.info(f"📅 Data coverage: 1980-2024 | 🦠 Diseases: {self.raw_data['Disease'].nunique()} | 🌍 Countries: {self.raw_data['Country / Region'].nunique()}")
                    return True
                else:
                    st.error("❌ Combined data file not found. Please run combine_disease_data.py first.")
                    return False
            except Exception as e:
                st.error(f"❌ Error loading combined data: {e}")
                return False

    def clean_numeric_value(self, value):
        """Clean and convert values to numeric"""
        if pd.isna(value):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace(' ', '').replace(',', '').replace('<', '').strip()
            if cleaned == '' or cleaned.lower() in ['na', 'nan', 'null']:
                return 0.0
            try:
                return float(cleaned)
            except:
                return 0.0
        return 0.0

    def prepare_data_for_analysis(self):
        """Prepare combined data for analysis"""
        with st.spinner('Preparing data for analysis...'):
            try:
                # Get year columns - they are already column names in the dataframe
                all_columns = self.raw_data.columns.tolist()
                year_columns = []
                
                for col in all_columns:
                    try:
                        # Check if column is a year (integer between 1980 and 2024)
                        year_val = int(col)
                        if 1980 <= year_val <= 2024:
                            year_columns.append(col)
                    except (ValueError, TypeError):
                        continue
                
                # Sort year columns
                year_columns = sorted(year_columns, key=lambda x: int(x))
                
                # Clean the disease names
                self.raw_data['Disease_Clean'] = self.raw_data['Disease'].str.replace('reported cases and incidence', '').str.strip()
                
                # Create analysis dataframe with cleaned data
                records = []
                
                for idx, row in self.raw_data.iterrows():
                    country = row['Country / Region']
                    disease = row['Disease_Clean']
                    
                    # Skip regional aggregations
                    if any(keyword in str(country).lower() for keyword in 
                           ['region', 'global', 'world', 'african', 'eastern', 'european', 'americas', 'asia', 'pacific']):
                        continue
                    
                    # Extract recent years (2015-2024) for focused analysis
                    recent_year_cols = [col for col in year_columns if int(col) >= 2015]
                    yearly_cases = {}
                    total_cases = 0
                    valid_years = 0
                    
                    for year_col in recent_year_cols:
                        case_val = self.clean_numeric_value(row[year_col])
                        yearly_cases[f'Year_{year_col}'] = case_val
                        total_cases += case_val
                        if case_val > 0:
                            valid_years += 1
                    
                    # Include if there's meaningful data
                    if valid_years >= 2 or total_cases > 0:
                        record = {
                            'Country': country,
                            'Disease': disease,
                            'Total_Cases': total_cases,
                            'Valid_Years': valid_years,
                            **yearly_cases
                        }
                        records.append(record)
                
                self.combined_df = pd.DataFrame(records)
                st.success(f"✅ Prepared {len(self.combined_df)} records for analysis")
                return True
                
            except Exception as e:
                st.error(f"Error preparing data: {e}")
                import traceback
                st.error(traceback.format_exc())
                return False

    def display_basic_statistics(self):
        """Display comprehensive statistics about the data"""
        st.header("📊 Basic Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(self.combined_df):,}")
        with col2:
            st.metric("Diseases", len(self.combined_df['Disease'].unique()))
        with col3:
            st.metric("Countries", len(self.combined_df['Country'].unique()))
        with col4:
            total_cases = self.combined_df['Total_Cases'].sum()
            st.metric("Total Cases (2015-2024)", f"{int(total_cases):,}")

        # Historical overview from raw data
        st.subheader("📅 Historical Overview (1980-2024)")
        
        # Get all year columns from raw data - they are column names, not indices
        all_columns = self.raw_data.columns.tolist()
        year_columns = []
        
        for col in all_columns:
            try:
                year_val = int(col)
                if 1980 <= year_val <= 2024:
                    year_columns.append(col)
            except (ValueError, TypeError):
                continue
        
        year_columns = sorted(year_columns, key=lambda x: int(x))
        
        # Calculate total cases per year across all diseases
        yearly_totals = []
        for year_col in year_columns:
            total = 0
            for val in self.raw_data[year_col]:
                total += self.clean_numeric_value(val)
            yearly_totals.append({'Year': int(year_col), 'Total_Cases': total})
        
        historical_df = pd.DataFrame(yearly_totals)
        
        fig = px.line(
            historical_df,
            x='Year',
            y='Total_Cases',
            title='Total Disease Cases Over Time (1980-2024) - All Diseases Combined',
            markers=True
        )
        fig.update_layout(
            height=400,
            xaxis_title='Year',
            yaxis_title='Total Cases',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Disease distribution
        st.subheader("📈 Cases by Disease (Recent Data: 2015-2024)")
        disease_stats = self.combined_df.groupby('Disease').agg({
            'Total_Cases': ['sum', 'mean', 'std', 'count']
        }).round(2)
        disease_stats.columns = ['Total Cases', 'Mean Cases', 'Std Dev', 'Records']
        disease_stats = disease_stats.sort_values('Total Cases', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                disease_stats.reset_index(),
                x='Disease',
                y='Total Cases',
                color='Total Cases',
                color_continuous_scale='Viridis',
                title='Total Cases by Disease (2015-2024)'
            )
            fig.update_layout(height=400)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(disease_stats.style.format({
                'Total Cases': '{:,.0f}',
                'Mean Cases': '{:,.2f}',
                'Std Dev': '{:,.2f}',
                'Records': '{:,.0f}'
            }), height=400)

        # Geographic distribution
        st.subheader("🌍 Geographic Distribution")
        country_stats = self.combined_df.groupby('Country').agg({
            'Total_Cases': 'sum',
            'Disease': 'count'
        }).sort_values('Total_Cases', ascending=False).head(20)
        country_stats.columns = ['Total Cases', 'Disease Count']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                country_stats.reset_index(),
                x='Total Cases',
                y='Country',
                orientation='h',
                color='Total Cases',
                color_continuous_scale='Reds',
                title='Top 20 Countries by Total Cases (2015-2024)'
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(country_stats.style.format({
                'Total Cases': '{:,.0f}',
                'Disease Count': '{:,.0f}'
            }), height=500)

        # Temporal trends for recent years
        st.subheader("📅 Recent Temporal Trends (2015-2024)")
        year_columns = [col for col in self.combined_df.columns if col.startswith('Year_')]
        yearly_totals = {}
        
        for year_col in year_columns:
            year = year_col.replace('Year_', '')
            yearly_totals[year] = self.combined_df[year_col].sum()
        
        temporal_df = pd.DataFrame(list(yearly_totals.items()), columns=['Year', 'Cases'])
        temporal_df = temporal_df.sort_values('Year')
        
        fig = px.line(
            temporal_df,
            x='Year',
            y='Cases',
            title='Total Cases Over Time - Recent Years (All Diseases)',
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Cases by disease over time
        st.subheader("📊 Disease-Specific Temporal Trends (2015-2024)")
        disease_temporal = []
        for disease in self.combined_df['Disease'].unique():
            disease_data = self.combined_df[self.combined_df['Disease'] == disease]
            for year_col in year_columns:
                year = year_col.replace('Year_', '')
                cases = disease_data[year_col].sum()
                disease_temporal.append({
                    'Disease': disease,
                    'Year': year,
                    'Cases': cases
                })
        
        disease_temporal_df = pd.DataFrame(disease_temporal)
        
        fig = px.line(
            disease_temporal_df,
            x='Year',
            y='Cases',
            color='Disease',
            title='Cases by Disease Over Time (2015-2024)',
            markers=True
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Historical disease comparison (from raw data)
        st.subheader("📊 Historical Disease Comparison (1980-2024)")
        
        # Get all historical year columns - they are column names
        all_columns = self.raw_data.columns.tolist()
        hist_year_cols = []
        
        for col in all_columns:
            try:
                year_val = int(col)
                if 1980 <= year_val <= 2024:
                    hist_year_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        hist_year_cols = sorted(hist_year_cols, key=lambda x: int(x))
        
        # Sample every 5 years for clearer visualization
        sampled_year_cols = [col for col in hist_year_cols if int(col) % 5 == 0 or col == hist_year_cols[-1]]
        
        disease_historical = []
        for idx, row in self.raw_data.iterrows():
            disease = row['Disease_Clean']
            country = row['Country / Region']
            
            # Only include actual diseases, not regional summaries
            if any(keyword in str(country).lower() for keyword in 
                   ['region', 'global', 'world', 'african', 'eastern', 'european', 'americas', 'asia', 'pacific']):
                continue
            
            for year_col in sampled_year_cols:
                cases = self.clean_numeric_value(row[year_col])
                disease_historical.append({
                    'Disease': disease,
                    'Year': str(year_col),
                    'Cases': cases
                })
        
        disease_hist_df = pd.DataFrame(disease_historical)
        disease_hist_agg = disease_hist_df.groupby(['Disease', 'Year'])['Cases'].sum().reset_index()
        
        # Get min and max years for the title
        sampled_years_int = [int(col) for col in sampled_year_cols]
        
        fig = px.line(
            disease_hist_agg,
            x='Year',
            y='Cases',
            color='Disease',
            title=f'Historical Disease Trends (Sampled Years: {min(sampled_years_int)}-{max(sampled_years_int)})',
            markers=True
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    def perform_pca_analysis(self):
        """Perform PCA analysis and visualization"""
        st.header("🔬 Principal Component Analysis (PCA)")
        
        # Prepare features for PCA
        year_columns = [col for col in self.combined_df.columns if col.startswith('Year_')]
        
        if len(year_columns) == 0:
            st.error("No year columns found for PCA analysis")
            return
        
        # Create feature matrix
        X = self.combined_df[year_columns].values
        
        # Handle NaN values - replace with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.features_scaled = X_scaled
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Store results
        self.pca_results = {
            'pca': pca,
            'X_pca': X_pca,
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }
        
        # Display PCA statistics
        st.subheader("📊 PCA Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Components", len(pca.components_))
        with col2:
            st.metric("PC1 Variance", f"{pca.explained_variance_ratio_[0]*100:.2f}%")
        with col3:
            st.metric("PC2 Variance", f"{pca.explained_variance_ratio_[1]*100:.2f}%")
        with col4:
            n_components_95 = np.argmax(self.pca_results['cumulative_variance'] >= 0.95) + 1
            st.metric("Components for 95% Variance", n_components_95)
        
        # Explained variance plot
        st.subheader("📈 Explained Variance by Component")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                y=pca.explained_variance_ratio_ * 100,
                name='Individual Variance',
                marker_color='steelblue'
            ))
            fig.update_layout(
                title='Variance Explained by Each Principal Component',
                xaxis_title='Principal Component',
                yaxis_title='Variance Explained (%)',
                height=400
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(self.pca_results['cumulative_variance']) + 1)),
                y=self.pca_results['cumulative_variance'] * 100,
                mode='lines+markers',
                name='Cumulative Variance',
                marker_color='darkred',
                line=dict(width=3)
            ))
            fig.add_hline(y=95, line_dash="dash", line_color="green", 
                         annotation_text="95% Threshold")
            fig.update_layout(
                title='Cumulative Variance Explained',
                xaxis_title='Number of Components',
                yaxis_title='Cumulative Variance (%)',
                height=400
            )
            st.plotly_chart(fig, width='stretch')
        
        # Component loadings
        st.subheader("🔍 Principal Component Loadings")
        
        # Create loadings dataframe
        n_components_to_show = min(5, len(pca.components_))
        loadings_df = pd.DataFrame(
            pca.components_[:n_components_to_show].T,
            columns=[f'PC{i+1}' for i in range(n_components_to_show)],
            index=[col.replace('Year_', '') for col in year_columns]
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Component Loadings (Year Contributions)**")
            st.dataframe(loadings_df.style.format('{:.4f}').background_gradient(cmap='RdYlGn', axis=None))
        
        with col2:
            fig = go.Figure(data=go.Heatmap(
                z=loadings_df.T.values,
                x=loadings_df.index,
                y=loadings_df.columns,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(
                title='Component Loadings Heatmap',
                xaxis_title='Year',
                yaxis_title='Principal Component',
                height=400
            )
            st.plotly_chart(fig, width='stretch')

    def visualize_pca_2d(self):
        """Visualize PCA in 2D"""
        st.subheader("🎯 PCA 2D Visualization")
        
        if self.pca_results is None:
            st.warning("Please run PCA analysis first")
            return
        
        X_pca = self.pca_results['X_pca']
        
        # Create visualization dataframe
        viz_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Disease': self.combined_df['Disease'],
            'Country': self.combined_df['Country'],
            'Total_Cases': self.combined_df['Total_Cases']
        })
        
        # Color by disease
        fig = px.scatter(
            viz_df,
            x='PC1',
            y='PC2',
            color='Disease',
            hover_data=['Country', 'Total_Cases'],
            title=f'PCA: First Two Principal Components (Colored by Disease)<br>'
                  f'PC1 explains {self.pca_results["explained_variance"][0]*100:.2f}%, '
                  f'PC2 explains {self.pca_results["explained_variance"][1]*100:.2f}%',
            labels={'PC1': f'PC1 ({self.pca_results["explained_variance"][0]*100:.2f}%)',
                    'PC2': f'PC2 ({self.pca_results["explained_variance"][1]*100:.2f}%)'},
            height=600
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig, width='stretch')
        
        # Size by total cases
        st.subheader("📊 PCA with Case Volume")
        fig = px.scatter(
            viz_df,
            x='PC1',
            y='PC2',
            color='Disease',
            size='Total_Cases',
            hover_data=['Country', 'Total_Cases'],
            title='PCA Visualization (Size by Total Cases)',
            labels={'PC1': f'PC1 ({self.pca_results["explained_variance"][0]*100:.2f}%)',
                    'PC2': f'PC2 ({self.pca_results["explained_variance"][1]*100:.2f}%)'},
            height=600
        )
        fig.update_traces(marker=dict(opacity=0.6, line=dict(width=0.5, color='white')))
        st.plotly_chart(fig, width='stretch')

    def visualize_pca_3d(self):
        """Visualize PCA in 3D"""
        st.subheader("🌐 PCA 3D Visualization")
        
        if self.pca_results is None:
            st.warning("Please run PCA analysis first")
            return
        
        X_pca = self.pca_results['X_pca']
        
        # Create 3D visualization
        viz_df = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'PC3': X_pca[:, 2],
            'Disease': self.combined_df['Disease'],
            'Country': self.combined_df['Country'],
            'Total_Cases': self.combined_df['Total_Cases']
        })
        
        fig = px.scatter_3d(
            viz_df,
            x='PC1',
            y='PC2',
            z='PC3',
            color='Disease',
            hover_data=['Country', 'Total_Cases'],
            title=f'PCA: First Three Principal Components<br>'
                  f'PC1: {self.pca_results["explained_variance"][0]*100:.2f}%, '
                  f'PC2: {self.pca_results["explained_variance"][1]*100:.2f}%, '
                  f'PC3: {self.pca_results["explained_variance"][2]*100:.2f}%',
            labels={'PC1': f'PC1 ({self.pca_results["explained_variance"][0]*100:.2f}%)',
                    'PC2': f'PC2 ({self.pca_results["explained_variance"][1]*100:.2f}%)',
                    'PC3': f'PC3 ({self.pca_results["explained_variance"][2]*100:.2f}%)'},
            height=700
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig, width='stretch')

    def correlation_analysis(self):
        """Perform correlation analysis"""
        st.header("🔗 Correlation Analysis")
        
        year_columns = [col for col in self.combined_df.columns if col.startswith('Year_')]
        
        # Calculate correlation matrix
        corr_matrix = self.combined_df[year_columns].corr()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Correlation heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=[col.replace('Year_', '') for col in year_columns],
                y=[col.replace('Year_', '') for col in year_columns],
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            fig.update_layout(
                title='Year-to-Year Correlation Matrix',
                xaxis_title='Year',
                yaxis_title='Year',
                height=600
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.write("**Correlation Statistics**")
            # Get upper triangle of correlation matrix
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            upper_tri = corr_matrix.where(mask)
            
            st.metric("Average Correlation", f"{upper_tri.mean().mean():.3f}")
            st.metric("Max Correlation", f"{upper_tri.max().max():.3f}")
            st.metric("Min Correlation", f"{upper_tri.min().min():.3f}")
            
            st.write("**Highly Correlated Years (r > 0.9)**")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        year_i = year_columns[i].replace('Year_', '')
                        year_j = year_columns[j].replace('Year_', '')
                        high_corr.append({
                            'Year 1': year_i,
                            'Year 2': year_j,
                            'Correlation': f"{corr_matrix.iloc[i, j]:.3f}"
                        })
            
            if high_corr:
                st.dataframe(pd.DataFrame(high_corr), hide_index=True)
            else:
                st.info("No year pairs with correlation > 0.9")

    def disease_clustering_analysis(self):
        """Analyze disease clustering patterns"""
        st.header("🎯 Disease Clustering Analysis")
        
        # Calculate disease-level features
        disease_features = []
        
        year_columns = [col for col in self.combined_df.columns if col.startswith('Year_')]
        
        for disease in self.combined_df['Disease'].unique():
            disease_data = self.combined_df[self.combined_df['Disease'] == disease]
            
            # Calculate aggregate statistics
            features = {
                'Disease': disease,
                'Total_Records': len(disease_data),
                'Mean_Cases': disease_data['Total_Cases'].mean(),
                'Median_Cases': disease_data['Total_Cases'].median(),
                'Std_Cases': disease_data['Total_Cases'].std(),
                'Max_Cases': disease_data['Total_Cases'].max(),
                'Total_All_Cases': disease_data['Total_Cases'].sum()
            }
            
            # Add yearly averages
            for year_col in year_columns:
                year = year_col.replace('Year_', '')
                features[f'Avg_{year}'] = disease_data[year_col].mean()
            
            disease_features.append(features)
        
        disease_df = pd.DataFrame(disease_features)
        
        # Display disease comparison
        st.subheader("📊 Disease Comparison")
        
        comparison_cols = ['Disease', 'Total_Records', 'Mean_Cases', 'Std_Cases', 'Total_All_Cases']
        comparison_df = disease_df[comparison_cols].sort_values('Total_All_Cases', ascending=False)
        
        st.dataframe(comparison_df.style.format({
            'Total_Records': '{:,.0f}',
            'Mean_Cases': '{:,.2f}',
            'Std_Cases': '{:,.2f}',
            'Total_All_Cases': '{:,.0f}'
        }), width='stretch')
        
        # Visualize disease patterns
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                self.combined_df,
                x='Disease',
                y='Total_Cases',
                title='Case Distribution by Disease',
                log_y=True
            )
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            fig = px.violin(
                self.combined_df,
                x='Disease',
                y='Total_Cases',
                title='Case Distribution (Violin Plot)',
                log_y=True
            )
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

    def advanced_visualizations(self):
        """Create advanced visualizations"""
        st.header("📈 Advanced Visualizations")
        
        # Disease comparison across top countries
        st.subheader("🌍 Disease Patterns in Top Countries")
        
        top_countries = self.combined_df.groupby('Country')['Total_Cases'].sum().nlargest(10).index
        top_country_data = self.combined_df[self.combined_df['Country'].isin(top_countries)]
        
        pivot_data = top_country_data.pivot_table(
            index='Country',
            columns='Disease',
            values='Total_Cases',
            aggfunc='sum',
            fill_value=0
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='YlOrRd',
            text=pivot_data.values,
            texttemplate='%{text:.0f}',
            textfont={"size": 9},
            colorbar=dict(title="Total Cases")
        ))
        fig.update_layout(
            title='Disease Cases Heatmap: Top 10 Countries',
            xaxis_title='Disease',
            yaxis_title='Country',
            height=500
        )
        st.plotly_chart(fig, width='stretch')
        
        # Stacked area chart
        st.subheader("📊 Temporal Evolution by Disease")
        
        year_columns = [col for col in self.combined_df.columns if col.startswith('Year_')]
        temporal_by_disease = []
        
        for year_col in sorted(year_columns):
            year = year_col.replace('Year_', '')
            for disease in self.combined_df['Disease'].unique():
                disease_data = self.combined_df[self.combined_df['Disease'] == disease]
                cases = disease_data[year_col].sum()
                temporal_by_disease.append({
                    'Year': year,
                    'Disease': disease,
                    'Cases': cases
                })
        
        temporal_df = pd.DataFrame(temporal_by_disease)
        
        fig = px.area(
            temporal_df,
            x='Year',
            y='Cases',
            color='Disease',
            title='Temporal Evolution of Disease Cases (Stacked Area)',
            height=500
        )
        st.plotly_chart(fig, width='stretch')

    def export_analysis_results(self):
        """Export analysis results"""
        st.header("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export combined data
            csv = self.combined_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Combined Data (CSV)",
                data=csv,
                file_name="disease_data_combined.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export PCA results
            if self.pca_results:
                pca_df = pd.DataFrame(
                    self.pca_results['X_pca'][:, :5],
                    columns=[f'PC{i+1}' for i in range(5)]
                )
                pca_df['Disease'] = self.combined_df['Disease'].values
                pca_df['Country'] = self.combined_df['Country'].values
                
                csv_pca = pca_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download PCA Results (CSV)",
                    data=csv_pca,
                    file_name="pca_results.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Export disease statistics
            disease_stats = self.combined_df.groupby('Disease').agg({
                'Total_Cases': ['sum', 'mean', 'std', 'count']
            })
            csv_stats = disease_stats.to_csv()
            st.download_button(
                label="📥 Download Statistics (CSV)",
                data=csv_stats,
                file_name="disease_statistics.csv",
                mime="text/csv"
            )

    def run(self):
        """Main application flow"""
        
        # Sidebar navigation
        st.sidebar.header("Navigation")
        analysis_sections = [
            "📊 Basic Statistics",
            "🔬 PCA Analysis",
            "🎯 PCA 2D Visualization",
            "🌐 PCA 3D Visualization",
            "🔗 Correlation Analysis",
            "🎯 Disease Clustering",
            "📈 Advanced Visualizations",
            "💾 Export Results"
        ]
        
        selected_sections = st.sidebar.multiselect(
            "Select Analysis Sections",
            analysis_sections,
            default=analysis_sections
        )
        
        # Load data
        if self.load_all_data():
            if self.prepare_data_for_analysis():
                
                # Display selected sections
                if "📊 Basic Statistics" in selected_sections:
                    self.display_basic_statistics()
                    st.markdown("---")
                
                if "🔬 PCA Analysis" in selected_sections:
                    self.perform_pca_analysis()
                    st.markdown("---")
                
                if "🎯 PCA 2D Visualization" in selected_sections:
                    if self.pca_results is None:
                        self.perform_pca_analysis()
                    self.visualize_pca_2d()
                    st.markdown("---")
                
                if "🌐 PCA 3D Visualization" in selected_sections:
                    if self.pca_results is None:
                        self.perform_pca_analysis()
                    self.visualize_pca_3d()
                    st.markdown("---")
                
                if "🔗 Correlation Analysis" in selected_sections:
                    self.correlation_analysis()
                    st.markdown("---")
                
                if "🎯 Disease Clustering" in selected_sections:
                    self.disease_clustering_analysis()
                    st.markdown("---")
                
                if "📈 Advanced Visualizations" in selected_sections:
                    self.advanced_visualizations()
                    st.markdown("---")
                
                if "💾 Export Results" in selected_sections:
                    self.export_analysis_results()
            else:
                st.error("Failed to prepare data for analysis")
        else:
            st.error("Failed to load data files")


if __name__ == "__main__":
    app = DiseaseDataVisualizer()
    app.run()
