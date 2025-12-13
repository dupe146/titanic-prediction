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

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/titanic-prediction.git
cd titanic-prediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the Flask application
python app.py
```

Then open your browser at: `http://127.0.0.1:5001`

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

## 💻 Usage

1. **Train the Model** (first time only):
```bash
   python train_model.py
```

2. **Start the Web Application**:
```bash
   python app.py
```

3. **Open Browser**: Navigate to `http://127.0.0.1:5001`

4. **Enter passenger details** and click "Predict Survival"

## 🎓 Features Used

The model uses 7 key features for prediction:
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Sex**: Gender (male/female)
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Ticket fare in British pounds
- **Embarked**: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

## 🛠️ Technologies Used

- **Backend**: Python, Flask
- **ML Library**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **Model**: Logistic Regression

## 📈 Model Details

- **Algorithm**: Logistic Regression
- **Training Data**: 80% of dataset (713 passengers)
- **Test Data**: 20% of dataset (178 passengers)
- **Features**: 7 (after preprocessing)
- **Target Variable**: Survived (0 = No, 1 = Yes)

## 🌐 Live Demo

https://titanic-prediction-wn6k.onrender.com/

## 👨‍💻 Author

Islamiat Modupeoluwa

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- Inspiration: Machine Learning Classification Project

---

**⭐ If you found this project helpful, please give it a star!**
