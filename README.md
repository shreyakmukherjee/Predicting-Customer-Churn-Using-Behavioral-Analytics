# Customer Churn Prediction Using Behavioral Analytics

This project aims to predict customer churn in a telecommunications company based on historical customer behavior. The goal is to identify customers who are at risk of leaving, allowing the company to take proactive measures to retain them. The project uses machine learning, specifically Support Vector Machines (SVM), to classify customers as either staying or churning.

## Problem Statement

A telecommunications company wants to reduce customer churn by identifying customers at risk of leaving. They have historical data on customer behavior and want to build a model to predict which customers are most likely to churn.

## Project Overview

The project uses a synthetic dataset that includes the following features:
- **Age**: The age of the customer.
- **MonthlyCharge**: The monthly charge the customer pays.
- **Churn**: The target variable indicating whether the customer churned (1) or stayed (0).

The goal is to predict the `churn` column based on the customer's `Age` and `MonthlyCharge` using a Support Vector Classifier (SVC).

## Features

- Data preprocessing and feature engineering using `pandas` and `numpy`.
- Implementation of Support Vector Machine (SVM) for churn prediction.
- Evaluation of model performance using accuracy and classification report metrics.
- User input interface to predict churn based on customer age and monthly charge.

## Getting Started

### Prerequisites

To run this project locally, you will need the following Python libraries:

- numpy
- pandas
- scikit-learn

You can install the necessary libraries using pip:

```bash
pip install numpy pandas scikit-learn
```

### Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```
2. Run the Python script:
```bash
python churn_prediction.py
```
3. Enter the customer details (age and monthly charge) when prompted to predict whether the customer is likely to churn or stay.

### Model Description
The model used in this project is a Support Vector Classifier (SVC), which is a supervised learning algorithm used for classification tasks. The model is trained on a dataset consisting of customer information (age and monthly charge) and the target variable churn.

1. Model Training
```bash
svc_model = SVC(kernel='linear', C=1.0)
svc_model.fit(X_train, y_train.values.flatten())
```
2. Model Prediction
The trained model is used to predict churn for a user-entered age and monthly charge.
```bash
prediction = svc_model.predict(user_input)
```
The output is a message indicating whether the customer is likely to stay or is at risk of churning.

### Results
The model’s performance can be evaluated using classification metrics such as accuracy, precision, recall, and F1-score.

### Contributing
Feel free to fork this repository, make improvements, and submit pull requests. If you find any issues or bugs, please open an issue.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
```bash

### Key Points:

- Replace the URL in the clone command with the actual URL of your GitHub repository.
- The `Model Description` section explains the model used, its training, and prediction steps.
- The `Usage` section provides clear instructions on how to run the script and input customer data for predictions.
```

