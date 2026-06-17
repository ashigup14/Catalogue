# International Football Match Result Prediction

This project uses machine learning to predict the results of international football matches based on historical data from 1872 to 2017.

## Dataset
Source: [International Football Results (1872-2017)](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

## Project Structure
```
├── data/
│   └── results.csv          # Kaggle dataset
├── models/
│   └── football_model.pkl   # Trained model
├── src/
│   ├── data_processor.py    # Data loading and preprocessing
│   ├── feature_engineer.py  # Feature engineering
│   ├── model_trainer.py     # Model training and evaluation
│   └── predictor.py         # Make predictions
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place `results.csv` in the `data/` folder

## Usage

### Train the Model
```bash
python main.py --train
```

### Make Predictions
```bash
python main.py --predict
```

## Model Details

- **Algorithm**: XGBoost Classifier
- **Features**: Team strength metrics, historical win rates, home/away advantage
- **Target**: Match outcome (Home Win, Draw, Away Win)

## Performance

The model achieves competitive accuracy on international match predictions by learning from over 140 years of football history.

## Author
Created by: ashigup14
