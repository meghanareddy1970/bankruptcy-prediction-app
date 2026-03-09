# Bankruptcy Prediction using Machine Learning

## Live Application
Open the deployed Streamlit application here:

[Open the Live App](https://bankruptcy-prediction-app-h6ydnc5dk92pwiiwkntcp3.streamlit.app/)

This application predicts the probability of company bankruptcy based on several financial and operational risk indicators.

---

## Project Overview
This project builds a machine learning model to predict whether a company is likely to go bankrupt.  
The model analyzes multiple financial risk indicators and estimates the probability of bankruptcy.

The project demonstrates a complete machine learning workflow including:

- Exploratory Data Analysis
- Model training and comparison
- Model evaluation
- Deployment as an interactive Streamlit web application

---

## Dataset Description

The dataset contains **250 companies** and **6 financial risk indicators** used to predict bankruptcy.

### Features
- Industrial Risk
- Management Risk
- Financial Flexibility
- Credibility
- Competitiveness
- Operating Risk

### Target Variable
0 → Non-bankruptcy  
1 → Bankruptcy  

Feature values represent risk levels:

0 → Low risk  
0.5 → Medium risk  
1 → High risk  

---

## Model

Several machine learning models were trained and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Logistic Regression was selected as the final model and integrated with a **StandardScaler pipeline**.

---

## Run the App Locally

After downloading or cloning the repository, run:

```
streamlit run app.py
```


---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- GitHub
- Streamlit Community Cloud

---

## Project Structure

```
bankruptcy-prediction-app
│
├── app.py
├── bankruptcy_model.pkl
├── requirements.txt
├── bankruptcy_analysis.ipynb
└── README.md
```

## Author
Meghana Annapureddy


