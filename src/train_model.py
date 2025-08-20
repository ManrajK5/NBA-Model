# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

# Load features
features_path = "data/features.csv"
df = pd.read_csv(features_path)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} cols from {features_path}")

# Features to use
feature_cols = [
    'HOME_GAME', 'PTS_rolling', 'REB_rolling', 'AST_rolling',
    'STL_rolling', 'BLK_rolling', 'TOV_rolling',
    'FG_PCT_rolling', 'FG3_PCT_rolling', 'FT_PCT_rolling'
]
target_col = "WIN"

X = df[feature_cols]
y = df[target_col]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
print("Training XGBoost…")
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=7,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(x_train, y_train)

# Evaluate simple accuracy
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"📊 Test Accuracy: {acc:.3f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")
print("💾 Saved trained model → models/xgb_model.pkl")
