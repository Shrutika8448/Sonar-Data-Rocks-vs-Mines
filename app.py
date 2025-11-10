import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from streamlit_extras.stylable_container import stylable_container

# --- Page setup ---
st.set_page_config(page_title="SONAR: Rock vs Mine", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
body {
    background-color: #f8faff;
}
.navbar {
    display: flex;
    justify-content: center;
    background-color: #004080;
    padding: 0.8rem 0;
    border-radius: 0 0 12px 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.nav-item {
    color: white;
    padding: 0.5rem 1.5rem;
    margin: 0 0.5rem;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
}
.nav-item:hover {
    background-color: #0066cc;
}
.nav-active {
    background-color: #0059b3;
}
.fade {
    animation: fadeEffect 0.6s;
}
@keyframes fadeEffect {
    from {opacity: 0;}
    to {opacity: 1;}
}
footer {
    text-align: center;
    padding: 1rem;
    background-color: #004080;
    color: white;
    border-radius: 12px 12px 0 0;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# --- Navbar logic ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

def set_tab(tab_name):
    st.session_state.active_tab = tab_name

navbar_html = f"""
<div class="navbar">
    <div class="nav-item {'nav-active' if st.session_state.active_tab=='Home' else ''}" onclick="window.location.href='#Home'">Home</div>
    <div class="nav-item {'nav-active' if st.session_state.active_tab=='Analysis' else ''}" onclick="window.location.href='#Analysis'">Analysis</div>
    <div class="nav-item {'nav-active' if st.session_state.active_tab=='Settings' else ''}" onclick="window.location.href='#Settings'">Settings</div>
</div>
"""
st.markdown(navbar_html, unsafe_allow_html=True)

# --- Navbar JS (simulate switching tabs dynamically) ---
st.markdown("""
<script>
const items = Array.from(document.querySelectorAll('.nav-item'));
items.forEach(item => item.addEventListener('click', e => {
    const text = e.target.innerText.trim();
    window.parent.postMessage({isStreamlitMessage:true, type:"SET_TAB", tab:text}, "*");
}));
</script>
""", unsafe_allow_html=True)

# Handle JS message
st.session_state.active_tab = st.experimental_get_query_params().get("tab", [st.session_state.active_tab])[0]

# --- Load model and dataset ---
@st.cache_data
def train_model():
    df = pd.read_csv("https://raw.githubusercontent.com/Shrutika8448/Sonar-Data-Rocks-vs-Mines/main/sonar.csv", header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, acc

model, scaler, acc = train_model()

# --- Main Content (Dynamic fade section) ---
st.markdown('<div class="fade">', unsafe_allow_html=True)

if st.session_state.active_tab == "Home":
    st.title("🛰️ SONAR: Rock vs Mine")
    st.write("This project predicts whether a given sonar signal represents a **rock** or a **mine** using machine learning.")
    col1, col2 = st.columns(2)

    with col1:
        st.image("https://raw.githubusercontent.com/Shrutika8448/Sonar-Data-Rocks-vs-Mines/main/rock.jpg", caption="Rock", use_container_width=True)
    with col2:
        st.image("https://raw.githubusercontent.com/Shrutika8448/Sonar-Data-Rocks-vs-Mines/main/mine.jpg", caption="Mine", use_container_width=True)

    st.subheader("🔍 Upload Dataset or Single Sample")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", df.head())

    st.write(f"Model accuracy: **{acc*100:.2f}%**")

elif st.session_state.active_tab == "Analysis":
    st.title("📊 Data Analysis & Insights")
    st.write("Here you can visualize various patterns from the sonar dataset.")
    st.line_chart(np.random.randn(20, 2))

elif st.session_state.active_tab == "Settings":
    st.title("⚙️ Environment & Settings")
    st.write("Model: Logistic Regression")
    st.write("Scaler: StandardScaler")
    st.write("Dataset Source: [GitHub Repo](https://github.com/Shrutika8448/Sonar-Data-Rocks-vs-Mines)")

st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<footer>
    Made with ❤️ using Streamlit | © 2025 SONAR Rock vs Mine
</footer>
""", unsafe_allow_html=True)
