"""
Exploratory Data Analysis Page
Interactive data visualization and statistics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="EDA - Wine Quality",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #8B0000;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #FFE5E5 0%, #FFFFFF 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
Comprehensive analysis of the wine quality dataset
</div>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    """Load the wine quality dataset"""
    try:
        # Try to load from backend data folder
        data_path = Path(__file__).parent.parent.parent.parent / 'backend' / 'data' / 'winequality-red.csv'
        df = pd.read_csv(data_path, sep=';')
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure winequality-red.csv is in backend/data/")
        return None

df = load_data()

if df is not None:
    # Sidebar filters
    with st.sidebar:
        st.header("🎛️ Filters")
        
        quality_filter = st.multiselect(
            "Filter by Quality",
            options=sorted(df['quality'].unique()),
            default=sorted(df['quality'].unique())
        )
        
        alcohol_range = st.slider(
            "Alcohol Content Range (%)",
            float(df['alcohol'].min()),
            float(df['alcohol'].max()),
            (float(df['alcohol'].min()), float(df['alcohol'].max()))
        )
    
    # Apply filters
    filtered_df = df[
        (df['quality'].isin(quality_filter)) &
        (df['alcohol'].between(alcohol_range[0], alcohol_range[1]))
    ]
    
    # Overview Section
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(filtered_df):,}")
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Quality Range", f"{filtered_df['quality'].min()}-{filtered_df['quality'].max()}")
    with col4:
        st.metric("Avg Quality", f"{filtered_df['quality'].mean():.2f}")
    
    # Display sample data
    st.subheader("📄 Sample Data")
    st.dataframe(filtered_df.head(10), use_container_width=True)
    
    # Statistical Summary
    st.subheader("📊 Statistical Summary")
    
    tab1, tab2 = st.tabs(["Descriptive Statistics", "Missing Values"])
    
    with tab1:
        st.dataframe(filtered_df.describe(), use_container_width=True)
    
    with tab2:
        missing_data = filtered_df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values in the dataset!")
        else:
            fig = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                labels={'x': 'Feature', 'y': 'Missing Count'},
                title='Missing Values by Feature'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Quality Distribution
    st.header("🎯 Quality Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart
        quality_counts = filtered_df['quality'].value_counts().sort_index()
        fig = px.bar(
            x=quality_counts.index,
            y=quality_counts.values,
            labels={'x': 'Quality Score', 'y': 'Count'},
            title='Wine Quality Distribution',
            color=quality_counts.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart
        fig = px.pie(
            values=quality_counts.values,
            names=quality_counts.index,
            title='Quality Score Proportions',
            color_discrete_sequence=px.colors.sequential.Reds
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature Distributions
    st.header("📈 Feature Distributions")
    
    # Select feature to visualize
    feature_cols = [col for col in df.columns if col != 'quality']
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_feature = st.selectbox(
            "Select Feature",
            feature_cols,
            index=feature_cols.index('alcohol')
        )
    
    with col2:
        plot_type = st.radio(
            "Plot Type",
            ["Histogram", "Box Plot", "Violin Plot"],
            horizontal=True
        )
    
    # Create plot based on selection
    if plot_type == "Histogram":
        fig = px.histogram(
            filtered_df,
            x=selected_feature,
            color='quality',
            marginal='box',
            title=f'{selected_feature.replace("_", " ").title()} Distribution by Quality',
            color_discrete_sequence=px.colors.sequential.Reds
        )
    elif plot_type == "Box Plot":
        fig = px.box(
            filtered_df,
            x='quality',
            y=selected_feature,
            color='quality',
            title=f'{selected_feature.replace("_", " ").title()} by Quality Score',
            color_discrete_sequence=px.colors.sequential.Reds
        )
    else:  # Violin Plot
        fig = px.violin(
            filtered_df,
            x='quality',
            y=selected_feature,
            color='quality',
            box=True,
            title=f'{selected_feature.replace("_", " ").title()} Distribution by Quality',
            color_discrete_sequence=px.colors.sequential.Reds
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis
    st.header("🔗 Correlation Analysis")
    
    # Calculate correlation matrix
    corr_matrix = filtered_df.corr()
    
    # Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Feature Correlation Heatmap',
        width=800,
        height=800,
        xaxis={'side': 'bottom'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top correlations with quality
    st.subheader("🎯 Features Most Correlated with Quality")
    
    quality_corr = corr_matrix['quality'].drop('quality').sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Positive Correlations** 🔼")
        positive_corr = quality_corr[quality_corr > 0].head(5)
        for feature, corr in positive_corr.items():
            st.metric(feature.replace('_', ' ').title(), f"{corr:.3f}")
    
    with col2:
        st.markdown("**Negative Correlations** 🔽")
        negative_corr = quality_corr[quality_corr < 0].tail(5)
        for feature, corr in negative_corr.items():
            st.metric(feature.replace('_', ' ').title(), f"{corr:.3f}")
    
    # Feature Relationships
    st.header("🔍 Feature Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("X-Axis Feature", feature_cols, index=feature_cols.index('alcohol'))
    
    with col2:
        y_feature = st.selectbox("Y-Axis Feature", feature_cols, index=feature_cols.index('volatile acidity'))
    
    # Scatter plot
    fig = px.scatter(
        filtered_df,
        x=x_feature,
        y=y_feature,
        color='quality',
        title=f'{x_feature.replace("_", " ").title()} vs {y_feature.replace("_", " ").title()}',
        color_continuous_scale='Reds',
        opacity=0.6
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    st.header("💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🍷 Alcohol Content**
        
        Higher alcohol content is strongly correlated with better wine quality. 
        Wines with >11% alcohol tend to score higher.
        """)
    
    with col2:
        st.info("""
        **🧪 Volatile Acidity**
        
        High volatile acidity (>0.6) is associated with poor quality wines. 
        It can lead to an unpleasant vinegar taste.
        """)
    
    with col3:
        st.info("""
        **⚖️ Balance is Key**
        
        The best wines maintain a balance between acidity, sugar, 
        sulfur dioxide, and alcohol content.
        """)
    
    # Download filtered data
    st.header("💾 Export Data")
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_wine_data.csv',
        mime='text/csv',
        use_container_width=True
    )

else:
    st.error("Unable to load dataset. Please check the file path.")