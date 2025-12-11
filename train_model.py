# train_model.py
# Model Training Script

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TITANIC SURVIVAL PREDICTION - MODEL TRAINING")
print("="*60)

# Load dataset (from data folder)
print("\n[1/6] Loading dataset...")
data = pd.read_csv('train.csv')
print(f"✓ Dataset loaded: {len(data)} passengers")

# Data preprocessing
print("\n[2/6] Preprocessing data...")
data_processed = data.copy()

# Drop unnecessary columns
data_processed = data_processed.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Handle missing values
data_processed['Age'] = data_processed['Age'].fillna(data_processed['Age'].median())
if data_processed['Embarked'].isnull().sum() > 0:
    data_processed['Embarked'].fillna(data_processed['Embarked'].mode()[0], inplace=True)

# Convert categorical to dummy variables
data_processed = pd.get_dummies(data_processed, columns=['Sex', 'Embarked'], drop_first=True)

print(f"✓ Data preprocessed: {data_processed.shape[1]-1} features")

# Split features and target
print("\n[3/6] Splitting data...")
X = data_processed.drop('Survived', axis=1)
y = data_processed['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Testing set: {len(X_test)} samples")

# Train model
print("\n[4/6] Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("✓ Model trained successfully!")

# Evaluate model
print("\n[5/6] Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"   • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   • Precision: {precision:.4f}")
print(f"   • Recall:    {recall:.4f}")
print(f"   • F1 Score:  {f1:.4f}")

# Save the model
print("\n[6/6] Saving model...")
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved as 'titanic_model.pkl'")

# Save feature columns
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("✓ Feature columns saved")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
print("\nYou can now run 'python app.py' to start the web application.")