"""Feature engineering module for football match prediction."""

import pandas as pd
import numpy as np
from typing import Tuple, List


class FeatureEngineer:
    """Create and engineer features for model training."""

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Initialize FeatureEngineer.
        
        Args:
            train_df: Training dataset
            test_df: Test dataset
        """
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None

    def create_team_strength_features(self, df: pd.DataFrame, cutoff_date=None) -> pd.DataFrame:
        """Create team strength features based on historical performance.
        
        Args:
            df: DataFrame to add features to
            cutoff_date: Date to calculate stats up to
            
        Returns:
            DataFrame with new features
        """
        df = df.copy()
        df['home_team_strength'] = 0.0
        df['away_team_strength'] = 0.0
        
        for idx, row in df.iterrows():
            # Use only historical data
            if cutoff_date:
                hist_df = self.train_df[self.train_df['date'] < row['date']]
            else:
                hist_df = self.train_df[self.train_df['date'] < row['date']]
            
            if len(hist_df) == 0:
                continue
            
            # Home team strength
            home_matches = hist_df[hist_df['home_team'] == row['home_team']]
            if len(home_matches) > 0:
                home_wins = len(home_matches[home_matches['outcome'] == 0])
                home_strength = home_wins / len(home_matches)
                df.at[idx, 'home_team_strength'] = home_strength
            
            # Away team strength
            away_matches = hist_df[hist_df['away_team'] == row['away_team']]
            if len(away_matches) > 0:
                away_wins = len(away_matches[away_matches['outcome'] == 2])
                away_strength = away_wins / len(away_matches)
                df.at[idx, 'away_team_strength'] = away_strength
        
        return df

    def create_recent_form_features(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Create features based on recent form (last N matches).
        
        Args:
            df: DataFrame to add features to
            window: Number of recent matches to consider
            
        Returns:
            DataFrame with recent form features
        """
        df = df.copy()
        df['home_recent_form'] = 0.0
        df['away_recent_form'] = 0.0
        
        for idx, row in df.iterrows():
            hist_df = self.train_df[self.train_df['date'] < row['date']].tail(window * 2)
            
            if len(hist_df) == 0:
                continue
            
            # Home team recent form
            home_recent = hist_df[hist_df['home_team'] == row['home_team']].tail(window)
            if len(home_recent) > 0:
                home_wins = len(home_recent[home_recent['outcome'] == 0])
                df.at[idx, 'home_recent_form'] = home_wins / len(home_recent)
            
            # Away team recent form
            away_recent = hist_df[hist_df['away_team'] == row['away_team']].tail(window)
            if len(away_recent) > 0:
                away_wins = len(away_recent[away_recent['outcome'] == 2])
                df.at[idx, 'away_recent_form'] = away_wins / len(away_recent)
        
        return df

    def create_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create head-to-head history features.
        
        Args:
            df: DataFrame to add features to
            
        Returns:
            DataFrame with h2h features
        """
        df = df.copy()
        df['h2h_home_wins'] = 0
        df['h2h_away_wins'] = 0
        df['h2h_draws'] = 0
        
        for idx, row in df.iterrows():
            hist_df = self.train_df[self.train_df['date'] < row['date']]
            
            # Find all previous matches between these teams
            h2h = hist_df[
                ((hist_df['home_team'] == row['home_team']) & (hist_df['away_team'] == row['away_team'])) |
                ((hist_df['home_team'] == row['away_team']) & (hist_df['away_team'] == row['home_team']))
            ]
            
            if len(h2h) > 0:
                # Count from home team perspective
                home_persp = h2h[h2h['home_team'] == row['home_team']]
                away_persp = h2h[h2h['away_team'] == row['home_team']]
                
                h2h_home_wins = len(home_persp[home_persp['outcome'] == 0]) + len(away_persp[away_persp['outcome'] == 2])
                h2h_away_wins = len(away_persp[away_persp['outcome'] == 0]) + len(home_persp[home_persp['outcome'] == 2])
                h2h_draws = len(home_persp[home_persp['outcome'] == 1]) + len(away_persp[away_persp['outcome'] == 1])
                
                df.at[idx, 'h2h_home_wins'] = h2h_home_wins
                df.at[idx, 'h2h_away_wins'] = h2h_away_wins
                df.at[idx, 'h2h_draws'] = h2h_draws
        
        return df

    def create_venue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create venue-related features.
        
        Args:
            df: DataFrame to add features to
            
        Returns:
            DataFrame with venue features
        """
        df = df.copy()
        df['is_neutral'] = df['neutral'].astype(int)
        return df

    def engineer_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute full feature engineering pipeline.
        
        Returns:
            Tuple of (train_X, train_y, test_X, test_y)
        """
        # Add features to training data
        train_features = self.train_df.copy()
        train_features = self.create_team_strength_features(train_features)
        train_features = self.create_recent_form_features(train_features)
        train_features = self.create_head_to_head_features(train_features)
        train_features = self.create_venue_features(train_features)
        
        # Add features to test data
        test_features = self.test_df.copy()
        test_features = self.create_team_strength_features(test_features)
        test_features = self.create_recent_form_features(test_features)
        test_features = self.create_head_to_head_features(test_features)
        test_features = self.create_venue_features(test_features)
        
        # Select features for the model
        feature_cols = [
            'home_team_strength',
            'away_team_strength',
            'home_recent_form',
            'away_recent_form',
            'h2h_home_wins',
            'h2h_away_wins',
            'h2h_draws',
            'is_neutral'
        ]
        
        self.train_X = train_features[feature_cols].fillna(0).values
        self.train_y = train_features['outcome'].values
        self.test_X = test_features[feature_cols].fillna(0).values
        self.test_y = test_features['outcome'].values
        
        print(f"Features created: {feature_cols}")
        print(f"Training set shape: {self.train_X.shape}")
        print(f"Test set shape: {self.test_X.shape}")
        
        return self.train_X, self.train_y, self.test_X, self.test_y

    def get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns:
            List of feature names
        """
        return [
            'home_team_strength',
            'away_team_strength',
            'home_recent_form',
            'away_recent_form',
            'h2h_home_wins',
            'h2h_away_wins',
            'h2h_draws',
            'is_neutral'
        ]
