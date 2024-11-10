# Spam Detector

This project implements a spam detection system for an Internet Service Provider (ISP). Using supervised machine learning techniques, the system analyzes email data and classifies messages as "spam" or "not spam." The classification accuracy is evaluated using two models: **Logistic Regression** and **Random Forest Classifier**.

## Table of Contents

- [Description](#description)
- [Code Structure](#code-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Additional Notes](#additional-notes)

## Description

The goal of this project is to develop a machine learning model capable of identifying spam emails. For this purpose, we use a dataset containing information on the frequency of certain words and characters in emails, as well as the length of capitalized sequences.

The project consists of the following stages:

1. **Data Loading and Preprocessing**: Pandas is used to import and visualize the dataset.
2. **Data Splitting**: The data is split into training and testing sets.
3. **Feature Scaling**: Numeric features are scaled to improve model performance.
4. **Model Training**: Two classification models are trained: **Logistic Regression** and **Random Forest Classifier**.
5. **Model Evaluation**: The accuracy of both models is compared to determine which is more effective.

## Code Structure

### 1. Importing Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
```

### 2. Loading Data

The dataset is imported from an external URL, and the first few rows are displayed to confirm successful loading.

```python
data = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv")
data.head()
```

### 3. Splitting Data into Training and Testing Sets

The labels (`y`) and features (`X`) are separated, and the data is split into training and testing sets.

```python
X = data.drop("spam", axis=1)
y = data["spam"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

### 4. Scaling Features

We use `StandardScaler` to scale the data, which helps improve the performance of certain models.

```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 5. Model Training and Evaluation

We train and evaluate two models: **Logistic Regression** and **Random Forest Classifier**. The accuracy of each model is calculated on the test set.

```python
# Logistic Regression
log_reg_model = LogisticRegression(random_state=1)
log_reg_model.fit(X_train_scaled, y_train)
log_reg_predictions = log_reg_model.predict(X_test_scaled)
log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_predictions)
```

### 6. Comparative Evaluation

The models are evaluated by comparing their accuracy scores to identify which model better classifies spam emails.

## Dependencies

To run this project, you need to install the following Python libraries:

```bash
pip install pandas scikit-learn
```

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/your_username/spam-detector.git
   cd spam-detector
   ```

2. Make sure the dependencies mentioned above are installed.

3. Run the Jupyter Notebook or `.py` file step by step in your preferred environment (such as Jupyter Notebook or VS Code).

4. Review the results and accuracy of each model at the end of the script.

## Results

In this project, both models achieved good accuracy, but the **Random Forest Classifier** outperformed **Logistic Regression** in terms of accuracy:

- **Logistic Regression Accuracy**: ~0.926
- **Random Forest Accuracy**: ~0.959

This result suggests that the Random Forest model is more effective in classifying emails as spam or not spam for this dataset.

## Additional Notes

- Choosing additional models or tuning hyperparameters could further improve accuracy.
- Although the Random Forest model achieved higher accuracy, its computational complexity is greater than Logistic Regression. It’s important to consider this when deploying a model in a production environment.
- The data used in this project comes from a public dataset in the UCI Machine Learning Library and is intended for educational purposes.

---
