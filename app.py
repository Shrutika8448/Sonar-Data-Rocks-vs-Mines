import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import time

st.set_page_config(page_title="SONAR: Rock vs Mine", layout="wide")

# --- Custom CSS for fade transitions and navbar ---
st.markdown("""
<style>
body {
    background-color: #ffffff;
    color: #000;
}
.navbar {
    display: flex;
    justify-content: center;
    gap: 30px;
    padding: 15px 0;
    background-color: #f8f9fa;
    border-bottom: 2px solid #007bff;
    font-size: 18px;
    font-weight: 600;
}
.navbar a {
    text-decoration: none;
    color: #007bff;
    transition: color 0.3s;
}
.navbar a:hover {
    color: #0056b3;
}
.fade-in {
    animation: fadeIn 0.8s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
.footer {
    margin-top: 50px;
    padding: 15px 0;
    text-align: center;
    font-size: 14px;
    color: #555;
    border-top: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# --- Navbar ---
st.markdown("""
<div class="navbar">
    <a href="?page=home">Home</a>
    <a href="?page=analysis">Analysis</a>
    <a href="?page=settings">Settings</a>
</div>
""", unsafe_allow_html=True)

# --- Get selected page ---
query_params = st.experimental_get_query_params()
page = query_params.get("page", ["home"])[0]

# --- Helper: Train model ---
@st.cache_resource
def train_model():
    df = pd.read_csv("https://raw.githubusercontent.com/ankurdome/sonar-dataset/main/sonar.csv", header=None)
    X = df.iloc[:, :-1]
    y = LabelEncoder().fit_transform(df.iloc[:, -1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    return model, scaler, acc

model, scaler, acc = train_model()

# --- PAGE: HOME ---
if page == "home":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("🪨 SONAR: Rock vs Mine")
    st.write("An intelligent system that classifies sonar signals as **Rock** or **Mine** using a KNN model trained on the classic Sonar dataset.")

    col1, col2 = st.columns(2)
    with col1:
        st.image("https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/sonar-images/main/rock.jpg", caption="Rock", use_container_width=True)
    with col2:
        st.image("https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/sonar-images/main/mine.jpg", caption="Mine", use_container_width=True)

    st.subheader("Upload a dataset or input a single sample:")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[1] == 61:
            st.success("✅ Labeled dataset detected.")
            counts = df.iloc[:, -1].value_counts()
            st.write(counts)
            fig, ax = plt.subplots()
            ax.bar(counts.index, counts.values)
            st.pyplot(fig)
        else:
            st.warning("Unlabeled dataset detected. Predicting labels...")
            preds = model.predict(scaler.transform(df))
            decoded = ["Mine" if p == 1 else "Rock" for p in preds]
            df["Prediction"] = decoded
            st.dataframe(df)
            fig, ax = plt.subplots()
            ax.bar(["Rock", "Mine"], [decoded.count("Rock"), decoded.count("Mine")])
            st.pyplot(fig)

    st.subheader("🔹 Single Sample Prediction")
    single_input = st.text_area("Enter 60 comma-separated values:")
    if st.button("Predict"):
        try:
            sample = np.array(single_input.split(","), dtype=float).reshape(1, -1)
            pred = model.predict(scaler.transform(sample))
            result = "Mine" if pred[0] == 1 else "Rock"
            st.success(f"Predicted: **{result}**")
        except Exception:
            st.error("Invalid input format. Please provide exactly 60 comma-separated values.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: ANALYSIS ---
elif page == "analysis":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("📊 Analysis")
    st.write("Explore patterns in the Sonar dataset and understand model predictions.")
    st.write(f"Model Accuracy: **{acc*100:.2f}%**")
    st.image("https://raw.githubusercontent.com/YOUR_GITHUB_USERNAME/sonar-images/main/analysis_graph.jpg", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- PAGE: SETTINGS ---
elif page == "settings":
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.title("⚙️ Settings & Environment")
    st.write("""
    **Environment Setup**
    - Python 3.12  
    - Streamlit  
    - scikit-learn  
    - pandas, numpy, matplotlib  
    """)
    st.info("You can update image URLs or dataset paths from your GitHub repo.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit | © 2025 SONAR Rock vs Mine
</div>
""", unsafe_allow_html=True)
