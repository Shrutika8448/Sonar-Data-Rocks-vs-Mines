import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="SONAR: Rock vs Mine", page_icon="🌊", layout="wide")

# Initialize session state
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Home"

# --------------------------------------------------
# TRAIN / LOAD MODEL
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("sonar.csv", header=None)
    return df


@st.cache_resource
def train_model():
    df = load_data()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    acc = model.score(X_scaled, y)
    return model, scaler, acc, df

model, scaler, acc, df = train_model()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
body {
  background-color: #f9f9f9;
  font-family: 'Inter', sans-serif;
}
.navbar {
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #ffffff;
  border-bottom: 3px solid #007BFF;
  padding: 1rem 0;
  gap: 4rem;
  font-weight: 600;
  font-size: 17px;
}
.nav-item {
  color: #333;
  cursor: pointer;
  position: relative;
  transition: color 0.3s ease-in-out;
}
.nav-item:hover {
  color: #007BFF;
}
.nav-item::after {
  content: '';
  position: absolute;
  width: 0%;
  height: 3px;
  left: 0;
  bottom: -5px;
  background-color: #007BFF;
  transition: width 0.3s ease-in-out;
  border-radius: 2px;
}
.nav-item:hover::after {
  width: 100%;
}
.nav-active {
  color: #007BFF;
}
.nav-active::after {
  width: 100%;
}
.footer {
  text-align: center;
  margin-top: 3rem;
  padding: 1rem;
  color: #555;
  font-size: 15px;
  border-top: 1px solid #ddd;
}
.fade {
  animation: fadeIn 0.5s ease-in;
}
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NAVBAR
# --------------------------------------------------
st.markdown('<div class="navbar">', unsafe_allow_html=True)

cols = st.columns([1, 1, 1])
tabs = ["Home", "Analysis", "Settings"]
icons = ["🏠", "📊", "⚙️"]

for i, tab in enumerate(tabs):
    if cols[i].button(f"{icons[i]} {tab}"):
        st.session_state.selected_tab = tab

st.markdown('</div>', unsafe_allow_html=True)

selected_tab = st.session_state.selected_tab

# --------------------------------------------------
# PAGE: HOME
# --------------------------------------------------
if selected_tab == "Home":
    st.markdown('<div class="fade">', unsafe_allow_html=True)
    st.title("🌊 SONAR: Rock vs Mine")
    st.write("Upload a dataset or single sample to classify between **Rock** and **Mine** signals using a logistic regression model.")

    col1, col2 = st.columns(2)
    with col1:
        st.image("./rock.jpg", caption="Rock", use_container_width=True)
    with col2:
        st.image("./mine.jpg", caption="Mine", use_container_width=True)

    st.subheader("🔹 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file, header=None)
        st.write("Uploaded Data Preview:")
        st.dataframe(data.head())

        if data.shape[1] == 61:  # labeled
            counts = data[60].value_counts()
            st.write("### Class Distribution")
            st.bar_chart(counts)
        else:  # unlabeled
            preds = model.predict(scaler.transform(data))
            counts = pd.Series(preds).value_counts()
            st.write("### Predicted Class Distribution")
            st.bar_chart(counts)
            st.write("### Predictions per Sample")
            st.write(preds)

    st.subheader("🔹 Single Sample Input")
    example = "0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032"
    sample_input = st.text_input("Enter comma-separated sample data (60 values):", example)

    if st.button("Predict Sample"):
        try:
            arr = np.array([float(x) for x in sample_input.split(",")]).reshape(1, -1)
            pred = model.predict(scaler.transform(arr))[0]
            st.success(f"### 🧭 Model Prediction: **{pred.upper()}**")
        except Exception as e:
            st.error("Invalid input. Please enter 60 comma-separated numeric values.")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# PAGE: ANALYSIS
# --------------------------------------------------
elif selected_tab == "Analysis":
    st.markdown('<div class="fade">', unsafe_allow_html=True)
    st.title("📊 Analysis Dashboard")

    counts = df[60].value_counts()
    st.subheader("Dataset Distribution (Rock vs Mine)")
    st.bar_chart(counts)

    st.subheader("Model Accuracy")
    st.metric(label="Training Accuracy", value=f"{acc*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# PAGE: SETTINGS
# --------------------------------------------------
elif selected_tab == "Settings":
    st.markdown('<div class="fade">', unsafe_allow_html=True)
    st.title("⚙️ Settings & Environment")
    st.write("This section contains environment details and setup steps.")
    st.code("""
    pip install -r requirements.txt
    streamlit run app.py
    """)
    st.write("Model: Logistic Regression")
    st.write("Scaler: StandardScaler")
    st.write("Dataset: sonar.csv (local / GitHub)")
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    <b>🌊 SONAR: Rock vs Mine</b> | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
