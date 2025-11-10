import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Rock vs Mine Classifier", layout="centered")

st.title("🎯 Rock vs Mine Classifier (Sonar Data)")
st.write("Upload a dataset or test a single sample to classify as **Rock** or **Mine**.")

# --- Upload training dataset ---
st.sidebar.header("📁 Model Training Data")
train_file = st.sidebar.file_uploader("Upload training dataset (with labels R/M)", type=["csv"])

@st.cache_resource
def train_model(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Fine-tuned KNN model
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='minkowski')
    knn.fit(X_train, y_train)

    acc = knn.score(X_test, y_test)
    return knn, scaler, le, acc

if train_file:
    df = pd.read_csv(train_file, header=None)
    model, scaler, le, acc = train_model(df)
    st.success(f"✅ Model trained successfully with accuracy: {acc*100:.2f}%")

    # --- Mode selection ---
    mode = st.radio("Choose Mode:", ["Upload Dataset for Analysis", "Single Sample Input"])

    # 1️⃣ Upload Dataset Mode
    if mode == "Upload Dataset for Analysis":
        uploaded_file = st.file_uploader("Upload dataset (labeled or unlabeled)", type=["csv"])
        if uploaded_file:
            data = pd.read_csv(uploaded_file, header=None)
            st.write("### Preview:")
            st.dataframe(data.head())

            # Check if labeled (R/M in last column)
            if data.iloc[:, -1].dtype == object or data.iloc[:, -1].isin(['R', 'M']).any():
                st.subheader("📊 Labeled Dataset Summary")
                label_counts = data.iloc[:, -1].value_counts()
                st.write(label_counts)

                fig, ax = plt.subplots()
                sns.barplot(x=label_counts.index, y=label_counts.values, palette="viridis", ax=ax)
                plt.title("Count of Rock vs Mine")
                st.pyplot(fig)

            else:
                # Predict for unlabeled dataset
                st.subheader("🧠 Predicting labels for uploaded samples...")
                X_new = scaler.transform(data)
                preds = model.predict(X_new)
                preds_labels = le.inverse_transform(preds)
                data['Predicted_Label'] = preds_labels
                st.dataframe(data.head())

                counts = pd.Series(preds_labels).value_counts()

                fig, ax = plt.subplots()
                sns.barplot(x=counts.index, y=counts.values, palette="coolwarm", ax=ax)
                plt.title("Predicted Count of Rock vs Mine")
                st.pyplot(fig)

                st.success(f"Total Samples: {len(preds_labels)} | Rocks: {counts.get('R',0)} | Mines: {counts.get('M',0)}")

                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predicted CSV", csv, "predicted_output.csv", "text/csv")

    # 2️⃣ Single Sample Input
    elif mode == "Single Sample Input":
        st.write("Enter 60 feature values separated by commas (from sonar sensor).")
        user_input = st.text_area("Sample Input", placeholder="0.0200, 0.0371, 0.0428, ... up to 60 values")

        if st.button("Predict"):
            try:
                values = [float(x.strip()) for x in user_input.split(",")]
                if len(values) != 60:
                    st.error("Please enter exactly 60 values!")
                else:
                    sample_scaled = scaler.transform([values])
                    pred = model.predict(sample_scaled)
                    label = le.inverse_transform(pred)[0]
                    st.success(f"🪨 The sample is predicted as: **{'Rock' if label=='R' else 'Mine'}**")
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.warning("👆 Please upload a labeled sonar dataset (with R/M in last column) first to train the model.")


