"""
Premium Wine Quality Predictor - Streamlit Frontend
Interactive dashboard for wine quality prediction
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

# Page configuration
# WHY: Set browser tab title and layout
# WHAT: Configure Streamlit page settings
# HOW: Use st.set_page_config at the very top
st.set_page_config(
    page_title="🍷 Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# WHY: Enhance visual appeal beyond default Streamlit
# WHAT: Apply custom CSS styling
# HOW: Use st.markdown with unsafe_allow_html
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #8B0000;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #FFE5E5 0%, #FFFFFF 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #8B0000;
    }
    .stButton>button {
        background-color: #8B0000;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #A52A2A;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)


def check_backend_status():
    """
    Check if backend API is running
    
    WHY: Verify connection before making predictions
    WHAT: Send GET request to backend health endpoint
    HOW: Use requests library with error handling
    
    Returns:
        tuple: (is_online: bool, status_message: str)
    """
    try:
        response = requests.get(f"{FASTAPI_BACKEND_ENDPOINT}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, f"✅ Backend Online ({data.get('models_count', 0)} models loaded)"
        else:
            return False, f"⚠️ Backend responding with status {response.status_code}"
    except requests.ConnectionError:
        return False, "❌ Backend Offline - Cannot connect"
    except requests.Timeout:
        return False, "❌ Backend Offline - Timeout"
    except Exception as e:
        logger.error(f"Error checking backend: {e}")
        return False, f"❌ Backend Error: {str(e)}"


def create_gauge_chart(value, title, color):
    """
    Create a gauge chart for displaying confidence/probability
    
    WHY: Visual representation of numerical values is more intuitive
    WHAT: Create plotly gauge chart
    HOW: Use plotly.graph_objects.Indicator
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#FFE5E5'},
                {'range': [33, 66], 'color': '#FFD4D4'},
                {'range': [66, 100], 'color': '#FFC3C3'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_probability_bar_chart(probabilities):
    """
    Create horizontal bar chart for class probabilities
    
    WHY: Compare probabilities across classes visually
    WHAT: Horizontal bar chart with color coding
    HOW: Use plotly express
    """
    df = pd.DataFrame({
        'Quality': list(probabilities.keys()),
        'Probability': [v * 100 for v in probabilities.values()]
    })
    
    colors = {
        'Poor': '#FF6B6B',
        'Average': '#FFD93D',
        'Good': '#6BCF7F'
    }
    
    df['Color'] = df['Quality'].map(colors)
    
    fig = px.bar(
        df, 
        x='Probability', 
        y='Quality',
        orientation='h',
        text='Probability',
        color='Quality',
        color_discrete_map=colors,
        title='Prediction Probabilities'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        showlegend=False,
        height=300,
        xaxis_title="Probability (%)",
        yaxis_title="",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def main():
    """
    Main application logic
    
    WHY: Orchestrate the entire dashboard
    WHAT: Create UI components and handle interactions
    HOW: Use Streamlit components and API calls
    """
    
    # Header
    st.markdown('<div class="main-header">🍷 Premium Wine Quality Predictor</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Predict wine quality using advanced machine learning models based on physicochemical properties
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/wine.png", width=80)
        st.title("⚙️ Configuration")
        
        # Backend status
        st.subheader("Backend Status")
        is_online, status_msg = check_backend_status()
        
        if is_online:
            st.success(status_msg)
        else:
            st.error(status_msg)
            st.info("""
            **To start the backend:**
            ```bash
            cd backend/src
            uvicorn main:app --reload
            ```
            """)
        
        st.divider()
        
        # Model selection
        st.subheader("Model Selection")
        model_choice = st.selectbox(
            "Choose prediction model:",
            ["xgboost", "random_forest"],
            help="XGBoost typically performs better"
        )
        
        st.divider()
        
        # Input method selection
        st.subheader("Input Method")
        input_method = st.radio(
            "How would you like to input data?",
            ["Manual Sliders", "Upload JSON File"],
            help="Choose between manual input or file upload"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Wine Properties")
        
        if input_method == "Manual Sliders":
            # Manual input with sliders
            # WHY: Intuitive way to explore different wine properties
            # WHAT: Create sliders for each feature
            # HOW: Use st.slider for each property
            
            with st.expander("🔬 Acidity Properties", expanded=True):
                fixed_acidity = st.slider(
                    "Fixed Acidity (g/dm³)", 
                    4.0, 16.0, 7.4, 0.1,
                    help="Most acids involved with wine or fixed or nonvolatile"
                )
                volatile_acidity = st.slider(
                    "Volatile Acidity (g/dm³)", 
                    0.1, 1.6, 0.7, 0.01,
                    help="Amount of acetic acid in wine (too high leads to vinegar taste)"
                )
                citric_acid = st.slider(
                    "Citric Acid (g/dm³)", 
                    0.0, 1.0, 0.0, 0.01,
                    help="Adds 'freshness' and flavor to wines"
                )
                pH = st.slider(
                    "pH Level", 
                    2.7, 4.2, 3.3, 0.01,
                    help="Describes acidity/basicity (0-14 scale)"
                )
            
            with st.expander("🍬 Sugar & Minerals", expanded=True):
                residual_sugar = st.slider(
                    "Residual Sugar (g/dm³)", 
                    0.9, 16.0, 2.5, 0.1,
                    help="Amount of sugar remaining after fermentation"
                )
                chlorides = st.slider(
                    "Chlorides (g/dm³)", 
                    0.01, 0.62, 0.08, 0.001,
                    help="Amount of salt in the wine"
                )
                sulphates = st.slider(
                    "Sulphates (g/dm³)", 
                    0.3, 2.0, 0.65, 0.01,
                    help="Wine additive which can contribute to SO2 levels"
                )
            
            with st.expander("💨 Sulfur Dioxide", expanded=True):
                free_sulfur_dioxide = st.slider(
                    "Free SO₂ (mg/dm³)", 
                    1.0, 72.0, 15.0, 1.0,
                    help="Free form of SO2 prevents microbial growth"
                )
                total_sulfur_dioxide = st.slider(
                    "Total SO₂ (mg/dm³)", 
                    6.0, 289.0, 46.0, 1.0,
                    help="Amount of free and bound forms of SO2"
                )
            
            with st.expander("⚗️ Physical Properties", expanded=True):
                density = st.slider(
                    "Density (g/cm³)", 
                    0.990, 1.004, 0.996, 0.0001,
                    help="Density of water is close to 1 depending on percent alcohol and sugar"
                )
                alcohol = st.slider(
                    "Alcohol Content (%)", 
                    8.0, 15.0, 10.5, 0.1,
                    help="Percent alcohol content of the wine"
                )
            
            # Prepare input data
            input_data = {
                "fixed_acidity": fixed_acidity,
                "volatile_acidity": volatile_acidity,
                "citric_acid": citric_acid,
                "residual_sugar": residual_sugar,
                "chlorides": chlorides,
                "free_sulfur_dioxide": free_sulfur_dioxide,
                "total_sulfur_dioxide": total_sulfur_dioxide,
                "density": density,
                "pH": pH,
                "sulphates": sulphates,
                "alcohol": alcohol
            }
            
        else:
            # File upload method
            st.info("📁 Upload a JSON file with wine properties")
            
            uploaded_file = st.file_uploader(
                "Choose a JSON file",
                type=['json'],
                help="Upload a JSON file containing wine properties"
            )
            
            if uploaded_file is not None:
                try:
                    input_data = json.load(uploaded_file)
                    st.success("✅ File uploaded successfully!")
                    st.json(input_data)
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
                    input_data = None
            else:
                input_data = None
                st.warning("⚠️ Please upload a file to continue")
        
        # Predict button
        st.divider()
        predict_button = st.button(
            "🔮 Predict Wine Quality",
            use_container_width=True,
            disabled=not is_online or (input_method == "Upload JSON File" and input_data is None)
        )
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Results container
        results_placeholder = st.empty()
        
        if predict_button and input_data:
            with st.spinner("🔄 Analyzing wine properties..."):
                try:
                    # Make API request
                    # WHY: Get prediction from trained models
                    # WHAT: POST request to /predict endpoint
                    # HOW: Send JSON data and receive prediction
                    
                    response = requests.post(
                        f"{FASTAPI_BACKEND_ENDPOINT}/predict",
                        params={"model_name": model_choice},
                        json=input_data,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        prediction = response.json()
                        
                        # Display results with styling
                        with results_placeholder.container():
                            # Quality result
                            quality_class = prediction['quality_class']
                            quality_label = prediction['quality_label']
                            confidence = prediction['confidence']
                            
                            # Color coding
                            colors = {
                                0: "#FF6B6B",  # Poor - Red
                                1: "#FFD93D",  # Average - Yellow
                                2: "#6BCF7F"   # Good - Green
                            }
                            
                            result_color = colors[quality_class]
                            
                            st.markdown(f"""
                            <div style='background-color: {result_color}20; 
                                        padding: 2rem; 
                                        border-radius: 10px; 
                                        border-left: 5px solid {result_color};
                                        margin-bottom: 1rem;'>
                                <h2 style='color: {result_color}; margin: 0;'>
                                    {quality_label}
                                </h2>
                                <p style='font-size: 1.2rem; margin-top: 0.5rem;'>
                                    Confidence: {confidence*100:.1f}%
                                </p>
                                <p style='color: #666; margin-top: 0.5rem;'>
                                    Model: {prediction['model_used'].upper()}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence gauge
                            st.plotly_chart(
                                create_gauge_chart(confidence, "Prediction Confidence", result_color),
                                use_container_width=True
                            )
                            
                            # Probabilities
                            st.plotly_chart(
                                create_probability_bar_chart(prediction['probabilities']),
                                use_container_width=True
                            )
                            
                            # Recommendations
                            st.subheader("💡 Recommendations")
                            
                            if quality_class == 2:
                                st.success("""
                                **Excellent Wine! 🎉**
                                - This wine has exceptional quality characteristics
                                - Ready for premium market positioning
                                - Consider aging potential for even better results
                                """)
                            elif quality_class == 1:
                                st.info("""
                                **Good Wine Quality 👍**
                                - Solid wine with room for improvement
                                - Consider adjusting alcohol content or acidity
                                - May benefit from additional aging
                                """)
                            else:
                                st.warning("""
                                **Quality Improvement Needed 🔧**
                                - Review acidity levels (fixed, volatile, pH)
                                - Check sulfur dioxide balance
                                - Consider alcohol content adjustment
                                """)
                            
                            # Download results
                            st.divider()
                            result_json = json.dumps(prediction, indent=2)
                            st.download_button(
                                label="📥 Download Results (JSON)",
                                data=result_json,
                                file_name="wine_prediction_results.json",
                                mime="application/json",
                                use_container_width=True
                            )
                            
                    else:
                        st.error(f"❌ Prediction failed: {response.text}")
                        
                except requests.Timeout:
                    st.error("❌ Request timed out. Backend might be overloaded.")
                except requests.ConnectionError:
                    st.error("❌ Cannot connect to backend. Please ensure it's running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Prediction error: {e}", exc_info=True)
        
        elif not predict_button:
            # Show placeholder
            with results_placeholder.container():
                st.info("""
                👈 **Configure your wine properties** on the left and click the predict button to see results!
                
                **What you'll get:**
                - Quality classification (Poor/Average/Good)
                - Confidence score
                - Detailed probability breakdown
                - Actionable recommendations
                """)
    
    # Footer
    st.divider()
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.metric("Model Type", model_choice.replace('_', ' ').title())
    with footer_col2:
        st.metric("Backend Status", "Online" if is_online else "Offline")
    with footer_col3:
        st.metric("Features Used", "14")
    
    # Info section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        ### 🍷 Premium Wine Quality Predictor
        
        This application uses machine learning to predict wine quality based on physicochemical properties.
        
        **Features:**
        - Multiple ML models (XGBoost, Random Forest)
        - Real-time predictions via REST API
        - Interactive visualizations
        - Detailed probability analysis
        
        **Tech Stack:**
        - Frontend: Streamlit + Plotly
        - Backend: FastAPI
        - Models: Scikit-learn, XGBoost
        
        **Data Source:**
        UCI Machine Learning Repository - Wine Quality Dataset
        
        **Created by:** [Your Name]
        """)


if __name__ == "__main__":
    main()