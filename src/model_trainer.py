"""Model training and evaluation module."""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import os
from typing import Dict, Tuple


class ModelTrainer:
    """Train and evaluate XGBoost model for football match prediction."""

    def __init__(self, model_path: str = 'models/football_model.pkl'):
        """Initialize ModelTrainer.
        
        Args:
            model_path: Path to save trained model
        """
        self.model_path = model_path
        self.model = None
        self.metrics = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train XGBoost classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Training XGBoost model...")
        
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            verbosity=1,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train, verbose=False)
        print("Model training complete!")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print("\n=== Model Evaluation ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n=== Confusion Matrix ===")
        print(cm)
        
        return self.metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities

    def save_model(self) -> None:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self) -> None:
        """Load trained model from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def get_feature_importance(self, feature_names: list) -> Dict:
        """Get feature importance from trained model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        importance_scores = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importance_scores))
        
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\n=== Feature Importance ===")
        for feature, importance in sorted_importance:
            print(f"{feature}: {importance:.4f}")
        
        return feature_importance
