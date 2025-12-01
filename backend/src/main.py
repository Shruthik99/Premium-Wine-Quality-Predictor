"""
FastAPI Backend for Wine Quality Prediction
Serves ML models via REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wine Quality Prediction API",
    description="REST API for predicting wine quality using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
# WHY: Allow frontend (Streamlit) to make requests from different origin
# WHAT: Enable Cross-Origin Resource Sharing
# HOW: Add middleware with allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
MODELS_DIR = Path(__file__).parent.parent / 'models'
models = {}
scaler = None
feature_names = []


class WineFeatures(BaseModel):
    """
    Pydantic model for wine features
    
    WHY: Data validation and automatic API documentation
    WHAT: Define expected input schema with validation rules
    HOW: Use Pydantic BaseModel with Field validators
    """
    fixed_acidity: float = Field(..., ge=0, le=20, description="Fixed acidity (g/dm³)")
    volatile_acidity: float = Field(..., ge=0, le=2, description="Volatile acidity (g/dm³)")
    citric_acid: float = Field(..., ge=0, le=2, description="Citric acid (g/dm³)")
    residual_sugar: float = Field(..., ge=0, le=20, description="Residual sugar (g/dm³)")
    chlorides: float = Field(..., ge=0, le=1, description="Chlorides (g/dm³)")
    free_sulfur_dioxide: float = Field(..., ge=0, le=100, description="Free SO₂ (mg/dm³)")
    total_sulfur_dioxide: float = Field(..., ge=0, le=300, description="Total SO₂ (mg/dm³)")
    density: float = Field(..., ge=0.99, le=1.01, description="Density (g/cm³)")
    pH: float = Field(..., ge=2.5, le=4.5, description="pH level")
    sulphates: float = Field(..., ge=0, le=2, description="Sulphates (g/dm³)")
    alcohol: float = Field(..., ge=8, le=15, description="Alcohol content (%)")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """
    Response model for predictions
    """
    quality_class: int = Field(..., description="Predicted quality class (0: Poor, 1: Average, 2: Good)")
    quality_label: str = Field(..., description="Quality label")
    confidence: float = Field(..., description="Prediction confidence")
    model_used: str = Field(..., description="Model used for prediction")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")


def load_models():
    """
    Load trained models at startup
    
    WHY: Load models once at startup rather than per request for efficiency
    WHAT: Load pickled models and scaler from disk
    HOW: Use joblib to deserialize saved artifacts
    """
    global models, scaler, feature_names
    
    try:
        logger.info("Loading models...")
        
        # Load Random Forest
        rf_path = MODELS_DIR / 'wine_model_rf.pkl'
        if rf_path.exists():
            models['random_forest'] = joblib.load(rf_path)
            logger.info("✅ Random Forest model loaded")
        else:
            logger.warning(f"⚠️ Random Forest model not found at {rf_path}")
        
        # Load XGBoost
        xgb_path = MODELS_DIR / 'wine_model_xgb.pkl'
        if xgb_path.exists():
            models['xgboost'] = joblib.load(xgb_path)
            logger.info("✅ XGBoost model loaded")
        else:
            logger.warning(f"⚠️ XGBoost model not found at {xgb_path}")
        
        # Load scaler
        scaler_path = MODELS_DIR / 'scaler.pkl'
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info("✅ Scaler loaded")
        else:
            logger.warning(f"⚠️ Scaler not found at {scaler_path}")
        
        # Load feature names
        feature_path = MODELS_DIR / 'feature_names.pkl'
        if feature_path.exists():
            feature_names = joblib.load(feature_path)
            logger.info(f"✅ Feature names loaded: {len(feature_names)} features")
        
        if not models:
            raise ValueError("No models were loaded successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise


def engineer_features(features: WineFeatures) -> np.ndarray:
    """
    Apply feature engineering to input features
    
    WHY: Same transformations must be applied at inference as during training
    WHAT: Create derived features matching training pipeline
    HOW: Calculate feature combinations and ratios
    """
    # Convert to dict then to values in correct order
    base_features = [
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.pH,
        features.sulphates,
        features.alcohol
    ]
    
    # Engineer features (same as training)
    total_acidity = features.fixed_acidity + features.volatile_acidity
    free_sulfur_ratio = features.free_sulfur_dioxide / max(features.total_sulfur_dioxide, 1e-6)
    alcohol_density_ratio = features.alcohol / max(features.density, 1e-6)
    
    # Combine all features
    all_features = base_features + [total_acidity, free_sulfur_ratio, alcohol_density_ratio]
    
    return np.array(all_features).reshape(1, -1)


@app.on_event("startup")
async def startup_event():
    """
    Load models when API starts
    
    WHY: Ensure models are ready before accepting requests
    WHAT: Execute model loading at startup
    HOW: FastAPI startup event handler
    """
    load_models()


@app.get("/")
async def root():
    """
    Health check endpoint
    
    WHY: Verify API is running and responsive
    WHAT: Return simple status message
    HOW: GET request to root endpoint
    """
    return {
        "status": "online",
        "message": "Wine Quality Prediction API is running",
        "version": "1.0.0",
        "models_loaded": list(models.keys())
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check
    """
    return {
        "status": "healthy",
        "models_available": list(models.keys()),
        "models_count": len(models),
        "features_count": len(feature_names)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_wine_quality(
    features: WineFeatures,
    model_name: str = "xgboost"
):
    """
    Predict wine quality
    
    WHY: Main API endpoint for getting predictions
    WHAT: Accept wine features and return quality prediction
    HOW: Validate input → Engineer features → Scale → Predict → Return result
    
    Args:
        features: Wine physicochemical properties
        model_name: Which model to use (xgboost or random_forest)
    
    Returns:
        Prediction with quality class, label, and confidence
    """
    try:
        # Validate model selection
        if model_name not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not available. Choose from: {list(models.keys())}"
            )
        
        # Get selected model
        model = models[model_name]
        
        # Engineer features
        X = engineer_features(features)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Map prediction to label
        quality_labels = {
            0: "Poor Quality (≤5)",
            1: "Average Quality (6)",
            2: "Good Quality (≥7)"
        }
        
        quality_label = quality_labels[prediction]
        confidence = float(probabilities[prediction])
        
        # Prepare probability dictionary
        prob_dict = {
            "Poor": float(probabilities[0]),
            "Average": float(probabilities[1]),
            "Good": float(probabilities[2])
        }
        
        logger.info(f"Prediction made: {quality_label} with confidence {confidence:.2f}")
        
        return PredictionResponse(
            quality_class=int(prediction),
            quality_label=quality_label,
            confidence=confidence,
            model_used=model_name,
            probabilities=prob_dict
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model-info")
async def get_model_info():
    """
    Get information about loaded models
    
    WHY: Provide transparency about model capabilities
    WHAT: Return model metadata and statistics
    HOW: Query model attributes
    """
    info = {
        "available_models": list(models.keys()),
        "features": feature_names,
        "feature_count": len(feature_names),
        "quality_classes": {
            0: "Poor Quality (≤5)",
            1: "Average Quality (6)",
            2: "Good Quality (≥7)"
        }
    }
    
    return info


@app.post("/predict-batch")
async def predict_batch(
    features_list: List[WineFeatures],
    model_name: str = "xgboost"
):
    """
    Batch prediction endpoint
    
    WHY: Efficiently predict multiple samples at once
    WHAT: Accept list of wine features and return list of predictions
    HOW: Process all samples together
    """
    try:
        if model_name not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not available"
            )
        
        model = models[model_name]
        predictions = []
        
        # Process each sample
        for features in features_list:
            X = engineer_features(features)
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            probabilities = model.predict_proba(X_scaled)[0]
            
            quality_labels = {
                0: "Poor Quality (≤5)",
                1: "Average Quality (6)",
                2: "Good Quality (≥7)"
            }
            
            predictions.append({
                "quality_class": int(prediction),
                "quality_label": quality_labels[prediction],
                "confidence": float(probabilities[prediction])
            })
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "model_used": model_name
        }
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    # WHY: Allow running directly for development
    # WHAT: Start uvicorn server programmatically
    # HOW: Call uvicorn.run()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )