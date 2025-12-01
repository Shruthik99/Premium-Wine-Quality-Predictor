"""
Model Comparison Page
Compare performance metrics of different ML models
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Model Comparison",
    page_icon="🤖",
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
    .winner-badge {
        background-color: #FFD700;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 Model Performance Comparison</div>', 
            unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
Detailed comparison of machine learning models trained for wine quality prediction
</div>
""", unsafe_allow_html=True)

# Model Performance Data
# NOTE: Replace these with your actual training results
models_data = {
    'Model': ['XGBoost', 'Random Forest', 'Logistic Regression'],
    'Accuracy': [0.891, 0.873, 0.821],
    'Precision': [0.88, 0.85, 0.80],
    'Recall': [0.87, 0.84, 0.79],
    'F1-Score': [0.87, 0.84, 0.79],
    'Training Time (min)': [3.2, 5.1, 1.8],
    'Inference Time (ms)': [2.1, 5.3, 0.8]
}

df_models = pd.DataFrame(models_data)

# Overall Performance Section
st.header("📊 Overall Performance Metrics")

# Metrics comparison table
st.dataframe(
    df_models.style.highlight_max(
        subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        color='lightgreen'
    ).highlight_min(
        subset=['Training Time (min)', 'Inference Time (ms)'],
        color='lightblue'
    ),
    use_container_width=True
)

# Winner announcement
best_model = df_models.loc[df_models['Accuracy'].idxmax(), 'Model']
best_accuracy = df_models.loc[df_models['Accuracy'].idxmax(), 'Accuracy']

st.markdown(f"""
<div style='text-align: center; margin: 2rem 0;'>
    <h3>🏆 Best Performing Model</h3>
    <span class='winner-badge'>{best_model} - {best_accuracy*100:.1f}% Accuracy</span>
</div>
""", unsafe_allow_html=True)

# Performance Metrics Comparison Charts
st.header("📈 Performance Metrics Visualization")

col1, col2 = st.columns(2)

with col1:
    # Grouped bar chart for main metrics
    fig = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=df_models['Model'],
            y=df_models[metric],
            marker_color=colors[i],
            text=df_models[metric].round(3),
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Performance Metrics Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Radar chart
    fig = go.Figure()
    
    for idx, row in df_models.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title='Multi-Metric Performance Radar',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Speed Comparison
st.header("⚡ Speed & Efficiency")

col1, col2 = st.columns(2)

with col1:
    # Training time comparison
    fig = px.bar(
        df_models,
        x='Model',
        y='Training Time (min)',
        title='Training Time Comparison',
        color='Training Time (min)',
        color_continuous_scale='Reds',
        text='Training Time (min)'
    )
    fig.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Inference time comparison
    fig = px.bar(
        df_models,
        x='Model',
        y='Inference Time (ms)',
        title='Inference Time Comparison (Lower is Better)',
        color='Inference Time (ms)',
        color_continuous_scale='Blues',
        text='Inference Time (ms)'
    )
    fig.update_traces(texttemplate='%{text:.1f} ms', textposition='outside')
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

# Confusion Matrix Section
st.header("🎯 Confusion Matrix Analysis")

st.info("""
**Confusion Matrix** shows how well each model classifies wines into Poor (0), Average (1), and Good (2) quality.
Diagonal values (correct predictions) should be high, while off-diagonal values (mistakes) should be low.
""")

# Example confusion matrices (replace with your actual results)
confusion_matrices = {
    'XGBoost': np.array([[145, 12, 3], [8, 98, 15], [2, 10, 87]]),
    'Random Forest': np.array([[142, 15, 3], [10, 95, 16], [5, 12, 82]]),
    'Logistic Regression': np.array([[135, 20, 5], [15, 88, 18], [8, 18, 73]])
}

selected_model = st.selectbox(
    "Select Model to View Confusion Matrix",
    list(confusion_matrices.keys())
)

# Plot confusion matrix
fig = go.Figure(data=go.Heatmap(
    z=confusion_matrices[selected_model],
    x=['Poor (0)', 'Average (1)', 'Good (2)'],
    y=['Poor (0)', 'Average (1)', 'Good (2)'],
    colorscale='Reds',
    text=confusion_matrices[selected_model],
    texttemplate='%{text}',
    textfont={"size": 16},
    colorbar=dict(title="Count")
))

fig.update_layout(
    title=f'Confusion Matrix - {selected_model}',
    xaxis_title='Predicted Quality',
    yaxis_title='Actual Quality',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Feature Importance Comparison
st.header("🎯 Feature Importance")

st.markdown("""
Understanding which features each model considers most important for prediction.
""")

# Example feature importance (replace with your actual results)
feature_importance = pd.DataFrame({
    'Feature': ['Alcohol', 'Volatile Acidity', 'Sulphates', 'Citric Acid', 'Total SO2', 
                'Density', 'Chlorides', 'Fixed Acidity', 'pH', 'Residual Sugar', 'Free SO2'],
    'XGBoost': [0.23, 0.18, 0.12, 0.09, 0.11, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02],
    'Random Forest': [0.21, 0.17, 0.13, 0.10, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02]
})

# Plot feature importance
fig = go.Figure()

fig.add_trace(go.Bar(
    name='XGBoost',
    x=feature_importance['Feature'],
    y=feature_importance['XGBoost'],
    marker_color='#FF6B6B'
))

fig.add_trace(go.Bar(
    name='Random Forest',
    x=feature_importance['Feature'],
    y=feature_importance['Random Forest'],
    marker_color='#4ECDC4'
))

fig.update_layout(
    title='Feature Importance Comparison',
    xaxis_title='Feature',
    yaxis_title='Importance',
    barmode='group',
    height=500,
    xaxis={'tickangle': -45}
)

st.plotly_chart(fig, use_container_width=True)

# Model Recommendations
st.header("💡 Model Selection Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("""
    **🏆 Best for Accuracy**
    
    **XGBoost**
    - Highest accuracy (89.1%)
    - Best F1-score
    - Balanced performance
    - **Recommended for production**
    """)

with col2:
    st.info("""
    **⚡ Best for Speed**
    
    **Logistic Regression**
    - Fastest inference (0.8ms)
    - Quick training
    - Lower accuracy trade-off
    - Good for real-time systems
    """)

with col3:
    st.warning("""
    **🎯 Most Interpretable**
    
    **Random Forest**
    - Easy to understand
    - Feature importance clear
    - Slightly slower
    - Good for analysis
    """)

# Detailed Comparison
st.header("📋 Detailed Model Characteristics")

characteristics = pd.DataFrame({
    'Characteristic': [
        'Accuracy',
        'Training Speed',
        'Inference Speed',
        'Interpretability',
        'Overfitting Risk',
        'Hyperparameter Tuning',
        'Memory Usage',
        'Production Ready'
    ],
    'XGBoost': ['⭐⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐⭐'],
    'Random Forest': ['⭐⭐⭐⭐', '⭐⭐', '⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐', '⭐⭐', '⭐⭐⭐⭐'],
    'Logistic Regression': ['⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐']
})

st.dataframe(characteristics, use_container_width=True)

# Export comparison
st.header("💾 Export Comparison")

csv = df_models.to_csv(index=False)
st.download_button(
    label="📥 Download Model Comparison Data",
    data=csv,
    file_name='model_comparison.csv',
    mime='text/csv',
    use_container_width=True
)