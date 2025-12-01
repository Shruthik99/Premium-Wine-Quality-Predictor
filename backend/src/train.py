"""
Wine Quality Model Training Script
Trains multiple ML models for wine quality prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class WineQualityTrainer:
    """
    Comprehensive trainer for wine quality prediction models
    """
    
    def __init__(self, data_path: str):
        """Initialize trainer with data path"""
        self.data_path = Path(data_path)
        self.models_dir = Path(__file__).parent.parent / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        # Model placeholders
        self.rf_model = None
        self.xgb_model = None
        self.scaler = None
        
        # Data placeholders
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """
        Load wine quality dataset and perform preprocessing
        
        WHY: Clean data is crucial for model performance
        WHAT: Load CSV, handle missing values, engineer features
        HOW: Use pandas for data manipulation
        """
        print("📊 Loading dataset...")
        
        # Load data
        df = pd.read_csv(self.data_path, sep=';')
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nQuality distribution:\n{df['quality'].value_counts().sort_index()}")
        
        # Check for missing values
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Convert quality to binary classification (good wine: quality >= 7)
        # OR keep it as multi-class (we'll use multi-class here for more challenge)
        print("\n🔧 Preprocessing data...")
        
        # Feature engineering
        df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
        df['free_sulfur_ratio'] = df['free sulfur dioxide'] / df['total sulfur dioxide']
        df['alcohol_density_ratio'] = df['alcohol'] / df['density']
        
        # Handle any infinite values from division
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.mean(), inplace=True)
        
        # Separate features and target
        X = df.drop('quality', axis=1)
        y = df['quality']
        
        # Recode quality to make it more balanced (combine similar classes)
        # 3,4,5 -> 0 (Poor), 6 -> 1 (Average), 7,8,9 -> 2 (Good)
        y = y.apply(lambda x: 0 if x <= 5 else (1 if x == 6 else 2))
        
        print(f"\nRecoded quality distribution:\n{y.value_counts().sort_index()}")
        print("  0: Poor quality (≤5)")
        print("  1: Average quality (6)")
        print("  2: Good quality (≥7)")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Save feature names for later use
        self.feature_names = X.columns.tolist()
        
        print(f"\n✅ Data preprocessing complete!")
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
    def train_random_forest(self):
        """
        Train Random Forest classifier
        
        WHY: Ensemble method that handles non-linear relationships well
        WHAT: Train RF with hyperparameter tuning
        HOW: Use GridSearchCV for optimal parameters
        """
        print("\n🌲 Training Random Forest model...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Create base model
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search
        grid_search = GridSearchCV(
            rf_base, 
            param_grid, 
            cv=3, 
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model
        self.rf_model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate
        y_pred = self.rf_model.predict(self.X_test)
        
        print("\n📊 Random Forest Performance:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(self.y_test, y_pred, average='weighted'):.4f}")
        print(f"F1-Score: {f1_score(self.y_test, y_pred, average='weighted'):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Poor', 'Average', 'Good']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
    def train_xgboost(self):
        """
        Train XGBoost classifier
        
        WHY: Gradient boosting often achieves best performance
        WHAT: Train XGBoost with optimized parameters
        HOW: Use native XGBoost API with cross-validation
        """
        print("\n🚀 Training XGBoost model...")
        
        # Define parameters
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        # Train model
        self.xgb_model = xgb.XGBClassifier(**params)
        self.xgb_model.fit(
            self.X_train, 
            self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.xgb_model.predict(self.X_test)
        
        print("\n📊 XGBoost Performance:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall: {recall_score(self.y_test, y_pred, average='weighted'):.4f}")
        print(f"F1-Score: {f1_score(self.y_test, y_pred, average='weighted'):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Poor', 'Average', 'Good']))
        
    def save_models(self):
        """
        Save trained models and scaler
        
        WHY: Persistence allows models to be deployed without retraining
        WHAT: Serialize models using joblib
        HOW: Save to models directory
        """
        print("\n💾 Saving models...")
        
        # Save Random Forest
        rf_path = self.models_dir / 'wine_model_rf.pkl'
        joblib.dump(self.rf_model, rf_path)
        print(f"✅ Random Forest saved to: {rf_path}")
        
        # Save XGBoost
        xgb_path = self.models_dir / 'wine_model_xgb.pkl'
        joblib.dump(self.xgb_model, xgb_path)
        print(f"✅ XGBoost saved to: {xgb_path}")
        
        # Save scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved to: {scaler_path}")
        
        # Save feature names
        feature_path = self.models_dir / 'feature_names.pkl'
        joblib.dump(self.feature_names, feature_path)
        print(f"✅ Feature names saved to: {feature_path}")
        
    def run_full_pipeline(self):
        """
        Execute complete training pipeline
        
        WHY: Orchestrate all training steps in sequence
        WHAT: Load data → Train models → Save artifacts
        HOW: Call all methods in order
        """
        print("="*70)
        print("🍷 WINE QUALITY MODEL TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Load and preprocess
        self.load_and_preprocess_data()
        
        # Step 2: Train models
        self.train_random_forest()
        self.train_xgboost()
        
        # Step 3: Save everything
        self.save_models()
        
        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETE!")
        print("="*70)
        print("\n🎉 Your models are ready for deployment!")
        print("📍 Next steps:")
        print("  1. Start the FastAPI backend: uvicorn main:app --reload")
        print("  2. Start the Streamlit frontend: streamlit run Dashboard.py")


def download_dataset():
    """
    Download wine quality dataset if not present
    
    WHY: Automate data acquisition
    WHAT: Download from UCI ML repository
    HOW: Use pandas read_csv with URL
    """
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    data_path = data_dir / 'winequality-red.csv'
    
    if not data_path.exists():
        print("📥 Downloading wine quality dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        
        try:
            df = pd.read_csv(url, sep=';')
            df.to_csv(data_path, index=False, sep=';')
            print(f"✅ Dataset downloaded to: {data_path}")
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("Please download manually from:")
            print("https://archive.ics.uci.edu/ml/datasets/wine+quality")
            raise
    
    return data_path


if __name__ == "__main__":
    # Download dataset if needed
    data_path = download_dataset()
    
    # Initialize trainer
    trainer = WineQualityTrainer(data_path)
    
    # Run complete pipeline
    trainer.run_full_pipeline()