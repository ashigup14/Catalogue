"""Prediction module for making predictions on new matches."""

import pandas as pd
import numpy as np
from src.model_trainer import ModelTrainer
from typing import Dict, List


class FootballPredictor:
    """Make predictions for football matches."""

    OUTCOME_MAP = {
        0: 'Home Win',
        1: 'Draw',
        2: 'Away Win'
    }

    def __init__(self, model_path: str = 'models/football_model.pkl'):
        """Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained model
        """
        self.trainer = ModelTrainer(model_path)
        self.trainer.load_model()

    def predict_match(self, features: np.ndarray) -> Dict:
        """Predict outcome of a single match.
        
        Args:
            features: Feature vector for the match
            
        Returns:
            Dictionary with prediction and probabilities
        """
        prediction, probabilities = self.trainer.predict(features.reshape(1, -1))
        
        pred_class = prediction[0]
        probs = probabilities[0]
        
        result = {
            'prediction': self.OUTCOME_MAP[pred_class],
            'confidence': probs[pred_class],
            'probabilities': {
                'home_win': probs[0],
                'draw': probs[1],
                'away_win': probs[2]
            }
        }
        
        return result

    def predict_batch(self, features: np.ndarray) -> List[Dict]:
        """Predict outcomes for multiple matches.
        
        Args:
            features: Feature matrix for matches
            
        Returns:
            List of prediction dictionaries
        """
        predictions, probabilities = self.trainer.predict(features)
        
        results = []
        for i, pred in enumerate(predictions):
            probs = probabilities[i]
            result = {
                'prediction': self.OUTCOME_MAP[pred],
                'confidence': probs[pred],
                'probabilities': {
                    'home_win': probs[0],
                    'draw': probs[1],
                    'away_win': probs[2]
                }
            }
            results.append(result)
        
        return results

    def predict_with_teams(self, home_team: str, away_team: str, features: np.ndarray) -> Dict:
        """Predict match outcome with team names.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            features: Feature vector
            
        Returns:
            Dictionary with prediction and team information
        """
        prediction = self.predict_match(features)
        prediction['home_team'] = home_team
        prediction['away_team'] = away_team
        
        return prediction
