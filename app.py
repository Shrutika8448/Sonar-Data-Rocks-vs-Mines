import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Rock vs Mine Classifier", layout="wide")

# ---------- STYLING ----------
st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
        }
        .navbar {
            display: flex;
            justify-content: center;
            background-color: #0f172a;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .navbar a {
            color: white;
            text-decoration: none;
            padding: 0 20px;
            font-weight: bold;
            font-size: 18px;
        }
        .navbar a:hover {
            color: #38bdf8;
        }
        footer {
            text-align: center;
            margin-top: 30px;
            font-size: 14px;
            color: #475569;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- NAVBAR ----------
st.markdown("""
<div class="navbar">
    <a href="#home">Home</a>
    <a href="#upload">Upload</a>
    <a href="#predict">Predict</a>
    <a href="#about">About</a>
</div>
""", unsafe_allow_html=True)

# ---------- HEADER SECTION ----------
st.markdown('<h1 id="home" style="text-align:center; color:#1e293b;">🎯 Rock vs Mine Classifier</h1>', unsafe_allow_html=True)
st.write("""
This AI-powered classifier predicts whether sonar signals reflect **rocks** or **mines** under the sea.  
Upload your dataset or manually input sonar readings to test the model.
""")

# ---------- IMAGES ----------
col1, col2 = st.columns(2)
with col1:
    st.image("https://images.unsplash.com/photo-1602524205483-16b6c70e7a5a", caption="Rock", use_container_width=True)
with col2:
    st.image("https://images.unsplash.com/photo-1607434472254-7677bb694b59", caption="Mine (Underwater Explosive)", use_container_width=True)

st.divider()

# ---------- SIDEBAR FOR TRAINING ----------
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

    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='minkowski')
    knn.fit(X_train, y_train)

    acc = knn.score(X_test, y_test)
    return knn, scaler, le, acc

if train_file:
    df = pd.read_csv(train_file, header=None)
    model, scaler, le, acc = train_model(df)
    st.success(f"✅ Model trained successfully with accuracy: {acc*100:.2f}%")

    # ---------- MODE SELECTION ----------
    mode = st.radio("Choose Mode:", ["Upload Dataset for Analysis", "Single Sample Input"], horizontal=True)

    # ---------- UPLOAD MODE ----------
    if mode == "Upload Dataset for Analysis":
        st.markdown('<h2 id="upload" style="color:#0f172a;">📊 Upload Dataset</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload dataset (labeled or unlabeled)", type=["csv"])
        if uploaded_file:
            data = pd.read_csv(uploaded_file, header=None)
            st.write("### Preview:")
            st.dataframe(data.head())

            if data.iloc[:, -1].dtype == object or data.iloc[:, -1].isin(['R', 'M']).any():
                st.subheader("Labeled Dataset Summary")
                label_counts = data.iloc[:, -1].value_counts()
                st.bar_chart(label_counts)
            else:
                st.subheader("🧠 Predicting labels for uploaded samples...")
                X_new = scaler.transform(data)
                preds = model.predict(X_new)
                preds_labels = le.inverse_transform(preds)
                data['Predicted_Label'] = preds_labels
                st.dataframe(data.head())

                counts = pd.Series(preds_labels).value_counts()
                st.bar_chart(counts)

                st.success(f"Total Samples: {len(preds_labels)} | Rocks: {counts.get('R',0)} | Mines: {counts.get('M',0)}")

                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Predicted CSV", csv, "predicted_output.csv", "text/csv")

    # ---------- SINGLE SAMPLE MODE ----------
    elif mode == "Single Sample Input":
        st.markdown('<h2 id="predict" style="color:#0f172a;">🧾 Predict a Single Sample</h2>', unsafe_allow_html=True)
        st.write("Enter 60 sonar feature values separated by commas:")

        user_input = st.text_area("Example: 0.0200, 0.0371, 0.0428, 0.0207, ... (60 values total)")

        if st.button("Predict"):
            try:
                values = [float(x.strip()) for x in user_input.split(",")]
                if len(values) != 60:
                    st.error("Please enter exactly 60 values!")
                else:
                    sample_scaled = scaler.transform([values])
                    pred = model.predict(sample_scaled)
                    label = le.inverse_transform(pred)[0]
                    st.success(f"🪨 Prediction: **{'Rock' if label=='R' else 'Mine'}**")
                    if label == 'R':
                        st.image("https://images.unsplash.com/photo-1602524205483-16b6c70e7a5a", caption="Rock Detected", use_container_width=True)
                    else:
                        st.image("https://images.unsplash.com/photo-1607434472254-7677bb694b59", caption="Mine Detected", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.warning("👆 Please upload a labeled sonar dataset (with R/M in last column) first to train the model.")

# ---------- ABOUT SECTION ----------
st.markdown('<h2 id="about" style="color:#0f172a;">ℹ️ About This Project</h2>', unsafe_allow_html=True)
st.write("""
This project uses **K-Nearest Neighbors (KNN)** algorithm to classify sonar signals as **Rock** or **Mine**.  
The model is trained on the classic *Sonar Mines vs Rocks* dataset, originally from the UCI Machine Learning Repository.

**Technologies Used:**
- Python 🐍  
- Scikit-learn ⚙️  
- Streamlit 🌐  
- Seaborn & Matplotlib 📊
""")

# ---------- FOOTER ----------
st.markdown("""
<footer>
    © 2025 Rock vs Mine Classifier | Built with ❤️ using Streamlit
</footer>
""", unsafe_allow_html=True)
