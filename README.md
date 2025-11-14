# Bike Sharing Demand Prediction

This project predicts hourly bike-sharing demand using the UCI Bike Sharing Dataset. The notebook performs end-to-end data processing, feature engineering, model training, evaluation, and model saving.

## 1. Introduction

Bike-sharing systems produce detailed usage logs that can be used to understand and predict demand at different hours of the day. Accurate demand prediction helps improve fleet management, resource allocation, and operational efficiency.
This project builds machine learning models to forecast hourly rental counts using weather conditions, seasonality, and temporal features.

## 2. Objective

- To analyze the Bike Sharing Dataset and understand key factors influencing demand.
- To engineer time-based and weather-based features for improving prediction accuracy.
- To build regression models (Linear Regression, Random Forest, Gradient Boosting).
- To evaluate and compare models using RMSE, MAE, and R².
- To save the best-performing model for deployment.

## 3. Dataset Description

The dataset used is the UCI Bike Sharing Dataset, containing:
- hour.csv → hourly rental data
- day.csv → daily rental aggregates

Key attributes:
- season, yr, mnth, hr, holiday, weekday, workingday
- Weather features: temp, atemp, hum, windspeed
- Target variable: cnt (total bike rentals for the hour)
The notebook performs EDA, examining distributions and demand patterns across seasons, weather conditions, and time features.

## 4. Model Description
### 4.1 Linear Regression

- Baseline model
- Uses feature scaling
- Good for interpretability but limited for nonlinear relationships

### 4.2 Random Forest Regressor

- Ensemble of decision trees
- Captures nonlinear trends
- Performs well without feature scaling

### 4.3 Gradient Boosting Regressor

- Boosted ensemble model
- Strong performance on structured/tabular data
- Tuned with n_estimators=200 and learning_rate=0.1

All models were trained on an 80-20 train-test split.

## 5. Coding Explanation
### ✔ Data Loading
- Downloads and extracts UCI ZIP file
- Loads hour.csv into a DataFrame

### ✔ Exploratory Data Analysis
- Displays structure of the dataset
- Visualizes distributions, trends, and correlations

### ✔ Feature Engineering
- Extracts hour, month, weekday, and season features
- Converts datetime values
- Applies cyclical encoding to hour (sin_hour, cos_hour)

### ✔ Preprocessing
- Splits dataset into train/test
- Scales numerical features using StandardScaler

### ✔ Model Training
- Trains Linear Regression, Random Forest, and Gradient Boosting
- Stores predictions for comparison

### ✔ Evaluation

Evaluated using:
- RMSE
- MAE
- R² score

### ✔ Best Model Selection
- Compares RMSE values to determine best model
- Plots actual vs predicted values for first 200 test samples

### ✔ Model Saving
Saves trained models using joblib:
- linear_regression_model.pkl
- random_forest_model.pkl
- gradient_boosting_model.pkl

6. Performance Analysis

- Tree-based models (Random Forest and Gradient Boosting) generally outperform Linear Regression
- Gradient Boosting often produces the lowest RMSE
- Random Forest feature importance highlights key contributors such as:
  - temperature
  - hour of day
  - humidity
  - season

A line plot comparing actual vs predicted values visualizes model performance on unseen data.
