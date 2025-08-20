# NBA Game Outcome Prediction Model  

## Abstract  
This repository contains a machine learning pipeline for predicting NBA game outcomes using historical game data. The model leverages **Random Forest** and **XGBoost** classifiers with feature engineering (team statistics, rolling averages, opponent performance) to estimate win probabilities.  

The project demonstrates a complete ML workflow including **data preprocessing, model training, evaluation, and serialization**.  


## Project Structure  

```text
NBA-Model/
├── notebooks/
│   └── nba_model.ipynb        # Main training notebook
├── data/
│   └── nba_games.csv          # Cleaned dataset (not pushed if large)
├── models/
│   └── nba_model.pkl          # Trained model file
├── src/
│   ├── train_model.py         # Training script
│   └── model_analysis.py      # Model evaluation & feature importance
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```


## Dependencies  
- Python 3.8+  
- pandas, numpy  
- scikit-learn  
- xgboost  
- matplotlib, seaborn  
- joblib  


## Usage
1. Training the Model
Open the Jupyter notebook: jupyter notebook notebooks/nba_model.ipynb
Run all cells to:
- Load and preprocess data
- Engineer features (rolling averages, opponent stats, etc.)
- Train Random Forest & XGBoost classifiers
- Save the trained model

2. Making Predictions
Load a saved model and predict outcomes


## Future Work
Real-time predictions via Basketball Reference scraping

Integration of player-level advanced stats

Live dashboard for visualizing game outcomes




