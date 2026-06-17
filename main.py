#!/usr/bin/env python
"""Main script for training and running football match predictions."""

import argparse
import os
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import FootballPredictor


def train_model():
    """Train the football prediction model."""
    print("\n=== Starting Training Pipeline ===")
    
    # Load and process data
    print("\n1. Loading and processing data...")
    processor = DataProcessor('data/results.csv')
    processor.process_data()
    train_df, test_df = processor.get_train_test_split(test_size=0.2)
    
    # Engineer features
    print("\n2. Engineering features...")
    engineer = FeatureEngineer(train_df, test_df)
    X_train, y_train, X_test, y_test = engineer.engineer_features()
    
    # Train model
    print("\n3. Training model...")
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    
    # Evaluate model
    print("\n4. Evaluating model...")
    trainer.evaluate(X_test, y_test)
    
    # Feature importance
    print("\n5. Feature importance...")
    feature_names = engineer.get_feature_names()
    trainer.get_feature_importance(feature_names)
    
    # Save model
    print("\n6. Saving model...")
    trainer.save_model()
    
    print("\n=== Training Complete ===")


def make_predictions():
    """Make predictions using trained model."""
    print("\n=== Making Predictions ===")
    
    # Check if model exists
    if not os.path.exists('models/football_model.pkl'):
        print("Error: Trained model not found. Please train the model first using --train")
        return
    
    # Load predictor
    predictor = FootballPredictor()
    
    # Example predictions
    print("\nExample match predictions:")
    print("-" * 50)
    
    # Create sample feature vectors (in practice, these would come from new matches)
    # Features: [home_strength, away_strength, home_form, away_form, h2h_hw, h2h_aw, h2h_draws, neutral]
    sample_matches = [
        ([0.55, 0.48, 0.50, 0.45, 3, 2, 1, 0], 'Team A', 'Team B'),
        ([0.60, 0.50, 0.60, 0.40, 2, 3, 2, 0], 'Team C', 'Team D'),
        ([0.50, 0.50, 0.50, 0.50, 1, 1, 2, 1], 'Team E', 'Team F'),
    ]
    
    for features, home_team, away_team in sample_matches:
        import numpy as np
        result = predictor.predict_with_teams(home_team, away_team, np.array(features))
        print(f"\n{home_team} vs {away_team}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities:")
        print(f"    - Home Win: {result['probabilities']['home_win']:.2%}")
        print(f"    - Draw: {result['probabilities']['draw']:.2%}")
        print(f"    - Away Win: {result['probabilities']['away_win']:.2%}")
    
    print("\n=== Predictions Complete ===")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Football Match Prediction Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.predict:
        make_predictions()
    else:
        print("Football Match Prediction Model")
        print("\nUsage:")
        print("  python main.py --train      Train the model")
        print("  python main.py --predict    Make predictions")
        print("\nSetup:")
        print("  1. pip install -r requirements.txt")
        print("  2. Download results.csv from Kaggle")
        print("  3. Place in data/ folder")
        print("  4. python main.py --train")
        print("  5. python main.py --predict")


if __name__ == '__main__':
    main()
