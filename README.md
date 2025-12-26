# 🚢 Titanic Survival Prediction

A machine learning web application that predicts passenger survival on the Titanic using Logistic Regression and Flask.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![ML](https://img.shields.io/badge/ML-Logistic_Regression-orange.svg)

## 🎯 Project Overview

This project uses machine learning to predict whether a passenger would survive the Titanic disaster based on features like:
- Passenger Class
- Gender
- Age
- Fare
- Number of family members aboard
- Port of embarkation

## 📊 Model Performance

- **Accuracy**: ~80%
- **Algorithm**: Logistic Regression
- **Dataset**: 891 passengers from Kaggle Titanic dataset

### Prerequisites
- Python 3.8 or higher
- pip package manager

## How to Test the Application

### Option 1: Try the Live Demo (No Setup Required)
Visit the live application: 
https://titanic-survival-prediction-s0um.onrender.com

**Test Cases to Try:**

**High Survival Probability:**
- Class: 1st
- Gender: Female
- Age: 30
- Fare: £100
- SibSp: 0
- Parch: 0
- Embarked: Southampton
**Expected Result:** ✅ SURVIVED (~85-95%)

**Low Survival Probability:**
- Class: 3rd
- Gender: Male
- Age: 25
- Fare: £8
- SibSp: 0
- Parch: 0
- Embarked: Southampton
**Expected Result:** ❌ NOT SURVIVED (~10-20%)

### Option 2: Run Locally
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/titanic-prediction.git
cd titanic-prediction

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the application
python app.py

# Open browser
http://127.0.0.1:5001
```
## 📁 Project Structure
```
titanic-prediction/
├── app.py                  # Flask server
├── train_model.py          # Model training script
├── templates/
│   └── index.html         # Web interface
├── data/
│   └── train.csv          # Dataset
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## 🎓 Features Used

The model uses 7 key features for prediction:
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Sex**: Gender (male/female)
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Ticket fare in British pounds
- **Embarked**: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

## 📈 Model Details
- **Algorithm**: Logistic Regression
- **Training Data**: 80% of dataset (713 passengers)
- **Test Data**: 20% of dataset (178 passengers)
- **Features**: 7 (after preprocessing)
- **Target Variable**: Survived (0 = No, 1 = Yes)

## 🎯 Understanding the Project

### The Essence
This project demonstrates the **complete machine learning workflow**:

1. **Data Science:** Analyzing historical Titanic data to find patterns
2. **Machine Learning:** Training a model to predict outcomes
3. **Software Engineering:** Building a web interface to make the model accessible
4. **Deployment:** Hosting the application so anyone can use it

### Real-World Applications
The techniques used here apply to:
- 🏥 **Healthcare:** Predicting patient outcomes
- 💰 **Finance:** Credit risk assessment

## 📊 Model Insights
**Why ~80% accuracy?**
- Some passengers' fates were truly random (luck)
- Missing data (we don't know everyone's cabin location)
- Complex factors we can't capture (did they help others evacuate?)

**Key Predictors:**
1. **Gender** (strongest): Women had 74% survival rate vs men's 19%
2. **Passenger Class**: 1st class (63%) > 2nd (47%) > 3rd (24%)
3. **Age**: Children prioritized in evacuation
4. **Fare**: Higher fare = better cabin location = easier escape

**This mirrors real history:** "Women and children first" policy

## 👨‍💻 Author

Islamiat Modupeoluwa

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- Inspiration: Machine Learning Classification Project

