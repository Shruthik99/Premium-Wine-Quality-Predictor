### 🍷 Premium Wine Quality Predictor 

### 🎯 Project Overview
An end-to-end Machine Learning Operations (MLOps) project that predicts wine quality based on physicochemical properties. This project demonstrates industry-standard practices including model serving via REST API, interactive frontend, and comprehensive ML pipeline implementation.

### 🌟 Key Features
Multi-Model Comparison: Random Forest, XGBoost, and Neural Network models
RESTful API: FastAPI backend for scalable model serving
Interactive Dashboard: Streamlit frontend with real-time predictions
Feature Engineering: Advanced feature creation and selection
Model Explainability: SHAP values for model interpretation
Data Visualization: Comprehensive EDA with interactive plots
Docker Support: Containerized deployment ready
CI/CD Ready: GitHub Actions workflow included

### 📊 Dataset
Source: UCI Machine Learning Repository - Wine Quality Dataset
The dataset contains 11 physicochemical features:

Fixed acidity
Volatile acidity
Citric acid
Residual sugar
Chlorides
Free sulfur dioxide
Total sulfur dioxide
Density
pH
Sulphates
Alcohol

Target: Wine quality score (0-10)


### 🚀 Getting Started
Prerequisites

Python 3.8 or higher
pip package manager
Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/wine-quality-mlops.git
cd wine-quality-mlops
```

2. **Set up Backend**
```bash
cd backend
python -m venv backend_env

# Activate virtual environment
# Windows:
backend_env\Scripts\activate
# Linux/Mac:
source backend_env/bin/activate

pip install -r requirements.txt
```

3. **Train the models**
```bash
cd src
python train.py
```
This will download the dataset automatically and train both XGBoost and Random Forest models.

4. **Set up Frontend** (in a new terminal)
```bash
cd frontend
python -m venv frontend_env

# Activate virtual environment
# Windows:
frontend_env\Scripts\activate
# Linux/Mac:
source frontend_env/bin/activate

pip install -r requirements.txt
```

### Running the Application

1. **Start the Backend API** (Terminal 1)
```bash
cd backend/src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
✅ API will be available at: `http://localhost:8000`  
📚 API Documentation: `http://localhost:8000/docs`

2. **Start the Frontend** (Terminal 2)
```bash
cd frontend/src
streamlit run Dashboard.py
```
✅ Application will open at: `http://localhost:8501`

### Dashboard Overview

![Dashboard Home](assets/screenshots/dashboard_home.png)
*Complete dashboard interface with sidebar configuration, wine property inputs, and prediction results area*

The dashboard features:
- **Left Sidebar**: Backend status monitoring, model selection, and input method configuration
- **Main Area**: Wine property input sliders and real-time prediction results
- **Responsive Layout**: Two-column design for optimal user experience
  <img width="975" height="504" alt="image" src="https://github.com/user-attachments/assets/8f148b78-73a0-4ab6-9417-dd1eb3828463" />


  ### ✅ Backend Health Monitoring

![Backend Status](assets/screenshots/backend_status.png)
*Real-time backend health check showing 2 models successfully loaded (XGBoost and Random Forest)*

<img width="600" height="273" alt="image" src="https://github.com/user-attachments/assets/23c0cca7-9017-4246-a1db-0ca9c5da806f" />

The application continuously monitors backend connectivity and displays:
- ✅ Green indicator when FastAPI backend is online
- Model count verification
- Clear error messages if connection fails

<img width="600" height="273" alt="image" src="https://github.com/user-attachments/assets/bf054fc5-ccbb-4804-b047-6dafaba41842" />


---
### 🔌 API Endpoint Response

![API Response]
<img width="624" height="166" alt="image" src="https://github.com/user-attachments/assets/d0b5e13a-d0cd-4c3b-bd46-33ecfdb3cf41" />

*FastAPI root endpoint returning status and loaded models information*

**API Response Format:**
```json
{
  "status": "online",
  "message": "Wine Quality Prediction API is running",
  "version": "1.0.0",
  "models_loaded": ["random_forest", "xgboost"]
}
```
<img width="975" height="259" alt="image" src="https://github.com/user-attachments/assets/48d115eb-94f5-40c8-91fc-e98f7431869b" />

Access interactive API documentation at: http://localhost:8000/docs

#### Method 1: File Upload

![File Upload](assets/screenshots/file_upload.png)
*Drag-and-drop JSON file upload with 200MB limit*
<img width="624" height="107" alt="image" src="https://github.com/user-attachments/assets/031c694c-7b0d-4f92-a4d6-087f67abd43b" />

Users can upload JSON files containing wine properties:

**Sample Input (`sample_test.json`):**
```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

<img width="975" height="167" alt="image" src="https://github.com/user-attachments/assets/812010e5-7e4c-4755-9e86-2a0f9bf7e2da" />

The uploaded file is previewed before prediction, allowing users to verify data correctness.

#### Method 2: Manual Sliders

![Manual Sliders](assets/screenshots/manual_sliders.png)
*Interactive sliders organized by property categories (Acidity, Sugar & Minerals, Sulfur Dioxide, Physical Properties)*

Each slider includes:
- Appropriate value ranges based on dataset statistics
- Helpful tooltips explaining each property
- Real-time value updates
- Expandable sections for better organization
  <img width="1097" height="820" alt="image" src="https://github.com/user-attachments/assets/8cd4739d-906f-497d-8b9e-ca00d32f4a7e" />

  <img width="1104" height="495" alt="image" src="https://github.com/user-attachments/assets/567eb587-9381-4580-bea0-b2b05a5d7d43" />







