import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("🎵 Sonar Rock vs Mine Prediction using KNN")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("sonarall-data.csv", header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le

X, y, le = load_data()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Fine-tune KNN using GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 20)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)
best_k = grid.best_params_['n_neighbors']

# Train final model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Display performance
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.write(f"✅ **Model trained successfully!** (Best k = {best_k}, Accuracy = {acc:.2f})")

# User input section
st.subheader("🔢 Enter Sonar Readings (60 features)")
st.caption("Each value should be between 0 and 1")

# User inputs
input_values = []
cols = st.columns(3)
for i in range(60):
    with cols[i % 3]:
        val = st.number_input(f"Feature {i+1}", min_value=0.0, max_value=1.0, step=0.01, key=f"f{i}")
        input_values.append(val)

# Predict button
if st.button("🔍 Predict"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = knn.predict(input_scaled)[0]
    label = "Rock" if prediction == 1 else "Mine"
    st.success(f"🎯 The object is predicted to be: **{label}**")
