# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import streamlit as st

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Split the data into feature and target
X = data.drop(columns=['Class'])
Y = data['Class']

# Rebalance the data using SMOTE
smot = SMOTE(random_state=42)
X_res, Y_res = smot.fit_resample(X, Y)

# Split the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.2, random_state=42)

# Train the models
rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)

xgb_model = XGBClassifier()
xgb_model.fit(X_train, Y_train)

# Calculate accuracy for Random Forest
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(Y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

# Calculate accuracy for XGBoost
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(Y_test, xgb_predictions)
print(f"XGBoost Accuracy: {xgb_accuracy}")

# Streamlit app
st.title('Credit Card Fraud Detection')
st.write('Please input all required features with comma separated')

# Input from user
input_features = st.text_input('Enter the features (comma-separated)')

# Choose classifier
classifier = st.selectbox('Select Classifier', ['Random Forest', 'XGBoost'])

# Submit button
submit = st.button('Submit')

if submit:
    features = np.array(input_features.split(','), dtype=np.float64)

    if classifier == 'Random Forest':
        prediction = rf_model.predict(features.reshape(1, -1))
    else:
        prediction = xgb_model.predict(features.reshape(1, -1))

    if prediction[0] == 0:
        st.success('The transaction is legitimate!')
    else:
        st.error('The transaction is a fraud!')

