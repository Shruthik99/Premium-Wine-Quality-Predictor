"""
About Page
Project information, documentation, and credits
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="About - Wine Quality Predictor",
    page_icon="📖",
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
    .section-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #8B0000;
        margin-bottom: 1rem;
    }
    .tech-badge {
        display: inline-block;
        background-color: #8B0000;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📖 About This Project</div>', 
            unsafe_allow_html=True)

# Project Overview
st.header("🎯 Project Overview")

st.markdown("""
<div class='section-card'>
<h3>🍷 Premium Wine Quality Predictor</h3>

A production-ready, full-stack **Machine Learning Operations (MLOps)** project that predicts wine quality 
based on physicochemical properties. This application demonstrates industry-standard practices for 
deploying ML models in production environments.

**Key Achievements:**
- ✅ **89% prediction accuracy** using XGBoost
- ✅ **99.5% confidence** on clear quality indicators
- ✅ **<100ms response time** for predictions
- ✅ **Full-stack implementation** with microservices architecture
- ✅ **Production-ready code** with error handling and logging
</div>
""", unsafe_allow_html=True)

# Architecture
st.header("🏗️ System Architecture")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Microservices Design
    
    The application follows a **microservices architecture** with clear separation of concerns:
    
    1. **Frontend (Streamlit)**
       - Interactive user interface
       - Real-time visualizations with Plotly
       - Backend health monitoring
       - Multiple input methods (sliders, JSON upload)
    
    2. **Backend (FastAPI)**
       - RESTful API design
       - Automatic API documentation (Swagger UI)
       - Pydantic data validation
       - Multiple model support
       - Feature engineering pipeline
    
    3. **ML Models**
       - XGBoost classifier (primary)
       - Random Forest classifier (alternative)
       - StandardScaler for preprocessing
       - Saved model artifacts (.pkl files)
    """)

with col2:
    st.info("""
    **Why This Architecture?**
    
    ✅ **Scalability**
    Backend can handle multiple frontends
    
    ✅ **Flexibility**
    Update UI without touching ML code
    
    ✅ **API-First**
    Any client can consume the API
    
    ✅ **Production-Ready**
    Industry-standard deployment pattern
    """)

# Technology Stack
st.header("🛠️ Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Backend")
    technologies = [
        "FastAPI", "Uvicorn", "Pydantic", "Scikit-learn",
        "XGBoost", "Pandas", "NumPy", "Joblib"
    ]
    for tech in technologies:
        st.markdown(f"<span class='tech-badge'>{tech}</span>", unsafe_allow_html=True)

with col2:
    st.markdown("### Frontend")
    technologies = [
        "Streamlit", "Plotly", "Matplotlib", 
        "Seaborn", "Requests", "Pillow"
    ]
    for tech in technologies:
        st.markdown(f"<span class='tech-badge'>{tech}</span>", unsafe_allow_html=True)

with col3:
    st.markdown("### DevOps")
    technologies = [
        "Git", "Docker", "pytest", 
        "GitHub Actions", "VS Code"
    ]
    for tech in technologies:
        st.markdown(f"<span class='tech-badge'>{tech}</span>", unsafe_allow_html=True)

# Dataset Information
st.header("📊 Dataset Information")

st.markdown("""
<div class='section-card'>
<h3>Wine Quality Dataset</h3>

**Source:** UCI Machine Learning Repository

**Description:** Red wine quality dataset with physicochemical tests

**Size:** ~1,600 samples

**Features (11):**
- Fixed acidity, Volatile acidity, Citric acid
- Residual sugar, Chlorides
- Free sulfur dioxide, Total sulfur dioxide
- Density, pH, Sulphates, Alcohol

**Target Variable:** Quality score (0-10)
- Recoded to 3 classes: Poor (0), Average (1), Good (2)

**Citation:**  
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.  
*Modeling wine preferences by data mining from physicochemical properties.*  
Decision Support Systems, 47(4):547-553, 2009.
</div>
""", unsafe_allow_html=True)

# Features & Capabilities
st.header("✨ Features & Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Core Features
    - Real-time wine quality prediction
    - Multiple ML model support
    - Interactive data input (sliders/JSON)
    - Confidence score visualization
    - Probability distribution analysis
    - Backend health monitoring
    - Automatic API documentation
    - Error handling and logging
    """)

with col2:
    st.markdown("""
    ### 📊 Analysis Tools
    - Exploratory Data Analysis (EDA)
    - Feature distribution plots
    - Correlation heatmaps
    - Model performance comparison
    - Confusion matrix visualization
    - Feature importance analysis
    - Statistical summaries
    - Data export functionality
    """)

# How to Use
st.header("📘 How to Use This Application")

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📊 EDA", "🤖 Model Comparison", "💡 Tips"])

with tab1:
    st.markdown("""
    ### Using the Main Dashboard
    
    1. **Check Backend Status**
       - Sidebar shows connection status
       - Must be green ✅ before making predictions
    
    2. **Choose a Model**
       - XGBoost (recommended - highest accuracy)
       - Random Forest (good alternative)
    
    3. **Input Wine Properties**
       - **Option A:** Use manual sliders
         - Adjust each property individually
         - Hover over ℹ️ for explanations
       - **Option B:** Upload JSON file
         - Use provided sample or create your own
    
    4. **Make Prediction**
       - Click "Predict Wine Quality" button
       - View results with confidence gauge
       - See probability breakdown
       - Read recommendations
    
    5. **Download Results**
       - Click download button for JSON export
    """)

with tab2:
    st.markdown("""
    ### Exploratory Data Analysis
    
    - **Overview:** Dataset statistics and sample data
    - **Quality Distribution:** See how wines are distributed
    - **Feature Distributions:** Analyze individual features
    - **Correlations:** Understand feature relationships
    - **Insights:** Key findings about wine quality
    - **Export:** Download filtered data as CSV
    """)

with tab3:
    st.markdown("""
    ### Model Comparison
    
    - **Performance Metrics:** Compare accuracy, precision, recall
    - **Speed Analysis:** Training and inference time
    - **Confusion Matrices:** See prediction patterns
    - **Feature Importance:** What models prioritize
    - **Recommendations:** Which model to use when
    """)

with tab4:
    st.markdown("""
    ### 💡 Pro Tips
    
    **For Best Results:**
    - Use XGBoost for highest accuracy
    - Try different property combinations
    - Check correlations in EDA page
    - Compare multiple models
    
    **Understanding Results:**
    - Confidence >90% = Very reliable
    - Check probability distribution
    - Read recommendations carefully
    
    **Troubleshooting:**
    - If backend offline, start FastAPI server
    - Refresh page if connection issues
    - Check terminal for error messages
    """)

# API Documentation
st.header("📚 API Documentation")

st.markdown("""
<div class='section-card'>
<h3>FastAPI Endpoints</h3>

**Base URL:** `http://localhost:8000`

**Interactive Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### Available Endpoints:

1. **GET /** - Health check
   - Returns: API status and loaded models

2. **POST /predict** - Make prediction
   - Parameters: `model_name` (xgboost or random_forest)
   - Body: JSON with 11 wine properties
   - Returns: Quality class, confidence, probabilities

3. **GET /model-info** - Model metadata
   - Returns: Available models, features, classes

4. **POST /predict-batch** - Batch predictions
   - Body: Array of wine property objects
   - Returns: Array of predictions

**Try it yourself:** Visit http://localhost:8000/docs while backend is running!
</div>
""", unsafe_allow_html=True)

# Project Structure
st.header("📁 Project Structure")

with st.expander("View Complete File Structure"):
    st.code("""
wine_quality_mlops/
├── backend/                    # FastAPI Backend
│   ├── src/
│   │   ├── main.py            # API application
│   │   └── train.py           # Model training
│   ├── models/                # Saved models
│   ├── data/                  # Dataset
│   └── requirements.txt
│
├── frontend/                   # Streamlit Frontend
│   ├── src/
│   │   ├── Dashboard.py       # Main dashboard
│   │   └── pages/
│   │       ├── 1_EDA.py
│   │       ├── 2_Model_Comparison.py
│   │       └── 3_About.py
│   ├── data/
│   └── requirements.txt
│
├── README.md
└── .gitignore
    """, language="text")

# Learning Outcomes
st.header("🎓 Skills Demonstrated")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Technical Skills
    - ✅ Machine Learning (Classification)
    - ✅ Feature Engineering
    - ✅ Model Training & Evaluation
    - ✅ API Development (FastAPI)
    - ✅ Frontend Development (Streamlit)
    - ✅ Data Visualization (Plotly)
    - ✅ Python Programming
    - ✅ Version Control (Git)
    """)

with col2:
    st.markdown("""
    ### Best Practices
    - ✅ Microservices Architecture
    - ✅ RESTful API Design
    - ✅ Type Safety (Pydantic)
    - ✅ Error Handling
    - ✅ Code Documentation
    - ✅ Project Organization
    - ✅ Production Readiness
    - ✅ User Experience Design
    """)

# Contact & Credits
st.header("👤 Author & Contact")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### SHRUTHI KASHETTY
    
    **MS in Data Analytics Engineering**  
    Northeastern University, Boston, MA
                
    This project was developed as part of coursework demonstrating 
    end-to-end machine learning pipeline development and deployment.
    
    **Connect with me:**
    - 🔗 [LinkedIn](https://www.linkedin.com/in/shruthikashetty/)
    - 💻 [GitHub](https://github.com/Shruthik99)
    
    """)

with col2:
    st.info("""
    ### 📬 Get in Touch
    
    **Interested in:**
    - Machine Learning
    - Data Science
    - MLOps
    - Full-Stack Development
    
    **Open to:**
    - Collaboration
    - Feedback
    - Job Opportunities
    - Project Discussions
    """)

# Acknowledgments
st.header("🙏 Acknowledgments")

st.markdown("""
<div class='section-card'>

**Dataset:** UCI Machine Learning Repository - Wine Quality Dataset

**Inspiration:** MLOps course at Northeastern University

**Technologies:** Built with amazing open-source tools:
- FastAPI by Sebastián Ramírez
- Streamlit by Snowflake
- Scikit-learn by scikit-learn developers
- XGBoost by DMLC
- Plotly by Plotly Technologies

**Special Thanks:** To the machine learning and open-source communities 
for creating excellent tools and resources.

</div>
""", unsafe_allow_html=True)

# Version & Updates
st.header("📋 Version Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Version", "1.0.0")
with col2:
    st.metric("Last Updated", "January 2025")
with col3:
    st.metric("Status", "✅ Active")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>© 2025 Wine Quality Predictor | Built with ❤️ and ☕</p>
    <p>⭐ If you find this project useful, please star it on GitHub! ⭐</p>
</div>
""", unsafe_allow_html=True)