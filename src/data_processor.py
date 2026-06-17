"""Data loading and preprocessing module for football match data."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import os


class DataProcessor:
    """Handle loading and preprocessing of football match data."""

    def __init__(self, data_path: str = 'data/results.csv'):
        """Initialize DataProcessor with data path.
        
        Args:
            data_path: Path to the results.csv file from Kaggle
        """
        self.data_path = data_path
        self.df = None
        self.processed_df = None

    def load_data(self) -> pd.DataFrame:
        """Load football match data from CSV.
        
        Returns:
            DataFrame with match data
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} matches")
        print(f"Columns: {list(self.df.columns)}")
        return self.df

    def create_match_outcome(self) -> pd.DataFrame:
        """Create target variable for match outcome.
        
        Target classes:
        - 0: Home team wins
        - 1: Draw
        - 2: Away team wins
        """
        def determine_outcome(home_goals, away_goals):
            if home_goals > away_goals:
                return 0  # Home win
            elif home_goals == away_goals:
                return 1  # Draw
            else:
                return 2  # Away win
        
        self.df['outcome'] = self.df.apply(
            lambda row: determine_outcome(row['home_score'], row['away_score']),
            axis=1
        )
        return self.df

    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Drop rows with missing critical values
        self.df = self.df.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'])
        
        # Fill neutral venue with False if missing
        self.df['neutral'] = self.df['neutral'].fillna(False)
        
        print(f"Data shape after handling missing values: {self.df.shape}")
        return self.df

    def convert_date_column(self) -> pd.DataFrame:
        """Convert date column to datetime."""
        self.df['date'] = pd.to_datetime(self.df['date'])
        return self.df

    def process_data(self) -> pd.DataFrame:
        """Execute full preprocessing pipeline.
        
        Returns:
            Processed DataFrame
        """
        self.load_data()
        self.convert_date_column()
        self.handle_missing_values()
        self.create_match_outcome()
        self.processed_df = self.df.copy()
        return self.processed_df

    def get_train_test_split(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets based on time.
        
        Args:
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if self.processed_df is None:
            self.process_data()
        
        # Sort by date
        df_sorted = self.processed_df.sort_values('date').reset_index(drop=True)
        
        # Time-based split
        split_point = int(len(df_sorted) * (1 - test_size))
        train_df = df_sorted[:split_point]
        test_df = df_sorted[split_point:]
        
        print(f"Train set: {len(train_df)} matches")
        print(f"Test set: {len(test_df)} matches")
        
        return train_df, test_df

    def get_team_stats(self, df: pd.DataFrame, team: str) -> Dict:
        """Calculate team statistics.
        
        Args:
            df: DataFrame to calculate stats from
            team: Team name
            
        Returns:
            Dictionary with team statistics
        """
        home_matches = df[df['home_team'] == team]
        away_matches = df[df['away_team'] == team]
        
        # Home stats
        home_wins = len(home_matches[home_matches['outcome'] == 0])
        home_draws = len(home_matches[home_matches['outcome'] == 1])
        home_losses = len(home_matches[home_matches['outcome'] == 2])
        home_gf = home_matches['home_score'].sum()
        home_ga = home_matches['away_score'].sum()
        
        # Away stats
        away_wins = len(away_matches[away_matches['outcome'] == 2])
        away_draws = len(away_matches[away_matches['outcome'] == 1])
        away_losses = len(away_matches[away_matches['outcome'] == 0])
        away_gf = away_matches['away_score'].sum()
        away_ga = away_matches['home_score'].sum()
        
        total_matches = home_matches.shape[0] + away_matches.shape[0]
        
        return {
            'team': team,
            'total_matches': total_matches,
            'home_wins': home_wins,
            'away_wins': away_wins,
            'draws': home_draws + away_draws,
            'total_wins': home_wins + away_wins,
            'goals_for': home_gf + away_gf,
            'goals_against': home_ga + away_ga,
            'goal_difference': (home_gf + away_gf) - (home_ga + away_ga)
        }
