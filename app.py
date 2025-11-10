import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="SONAR: Rock vs Mine", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #f9fbfd;
    font-family: 'Poppins', sans-serif;
}
.navbar {
    background-color: #007BFF;
    padding: 1rem 0;
    border-radius: 10px;
    text-align: center;
}
.navbar a {
    text-decoration: none;
    color: white;
    padding: 0 20px;
    font-weight: 600;
    font-size: 17px;
}
.navbar a:hover {
    color: #e0e0e0;
}
footer {
    text-align: center;
    padding: 20px;
    color: #555;
    font-size: 15px;
    border-top: 1px solid #ddd;
    margin-top: 40px;
}
.title {
    text-align: center;
    color: #003366;
    margin-top: 20px;
    font-size: 40px;
    font-weight: 700;
}
.subtext {
    text-align: center;
    color: #555;
    font-size: 18px;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ NAVBAR ------------------
st.markdown("""
<div class="navbar">
    <a href="#home">Home</a>
    <a href="#analysis">Analysis</a>
    <a href="#settings">Settings</a>
</div>
""", unsafe_allow_html=True)

# ------------------ PAGE HEADING ------------------
st.markdown('<h1 id="home" class="title">SONAR: Rock vs Mine</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtext">AI-powered classification of sonar signals into Rocks and Mines</p>', unsafe_allow_html=True)

# ------------------ IMAGES ------------------
col1, col2 = st.columns(2)
with col1:
    st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=800&q=80",
             caption="Rock Surface", use_container_width=True)
with col2:
    st.image("https://images.unsplash.com/photo-1581066312925-7f87c2b9d635?auto=format&fit=crop&w=800&q=80",
             caption="Underwater Mine", use_container_width=True)

st.divider()

# ------------------ MODEL TRAINING ------------------
@st.cache_resource
def train_model(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    return knn, scaler, le, acc

# ------------------ FILE UPLOAD ------------------
st.subheader("📂 Upload Your Sonar Dataset (with or without labels)")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # If labeled, show counts
    if df.shape[1] == 61 or df.iloc[:, -1].isin(['R', 'M']).any():
        st.success("Detected labeled dataset ✅")
        label_counts = df.iloc[:, -1].value_counts()
        st.bar_chart(label_counts)
        st.write(f"Rocks: {label_counts.get('R', 0)} | Mines: {label_counts.get('M', 0)}")

        model, scaler, le, acc = train_model(df)
        st.info(f"Model trained with accuracy: {acc*100:.2f}%")

    else:
        st.info("No labels detected — using pre-trained model to predict...")
        st.warning("Please first train the model using a labeled dataset above.")

# ------------------ SINGLE SAMPLE INPUT ------------------
st.divider()
st.subheader("🔍 Predict a Single Sample")

example = "0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111, 0.1609, 0.1582, 0.2238, 0.0645, 0.0660, 0.2273, 0.3100, 0.2999, 0.5078, 0.4797, 0.5783, 0.5071, 0.4328, 0.5550, 0.6711, 0.6415, 0.7104, 0.8080, 0.6791, 0.3857, 0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744, 0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324, 0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167, 0.0180, 0.0084, 0.0090, 0.0032"

if "single_sample_input" not in st.session_state:
    st.session_state.single_sample_input = example

user_input = st.text_area("Enter 60 comma-separated values:", st.session_state.single_sample_input)

if st.button("Predict"):
    try:
        values = [float(x.strip()) for x in user_input.split(",")]
        if len(values) != 60:
            st.error("Please enter exactly 60 values.")
        else:
            if uploaded_file:
                model, scaler, le, _ = train_model(df)
                X_sample = scaler.transform([values])
                pred = model.predict(X_sample)
                label = le.inverse_transform(pred)[0]
                st.success(f"🎯 Prediction: **{'Rock 🪨' if label == 'R' else 'Mine 💣'}**")
            else:
                st.warning("Upload a labeled dataset first to train the model.")
    except Exception as e:
        st.error(f"Error: {e}")

# ------------------ ANALYSIS SECTION ------------------
st.divider()
st.markdown('<h2 id="analysis" style="color:#003366;">📈 Analysis Dashboard</h2>', unsafe_allow_html=True)
st.write("Visualize sonar data trends, feature distributions, and model behavior here (to be expanded).")

# ------------------ SETTINGS SECTION ------------------
st.divider()
st.markdown('<h2 id="settings" style="color:#003366;">⚙️ Settings & Environment</h2>', unsafe_allow_html=True)
st.write("""
**Environment Details:**
- Python 3.12  
- Libraries: pandas, numpy, scikit-learn, matplotlib, streamlit  
- Model: K-Nearest Neighbors (k=3, distance weighted)  
- Scaling: StandardScaler  
- Encoding: LabelEncoder  
""")

# ------------------ FOOTER ------------------
st.markdown("""
<footer>
    <p>© 2025 <b>SONAR: Rock vs Mine</b> | Built with ❤️ using Streamlit</p>
</footer>
""", unsafe_allow_html=True)
