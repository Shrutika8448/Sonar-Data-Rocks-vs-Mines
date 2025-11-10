import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="SONAR: Rock vs Mine", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: white; color: #1a1a1a; }
    .navbar {
        display: flex; justify-content: center; gap: 2rem; 
        padding: 1rem; background-color: #007BFF; border-radius: 0 0 10px 10px;
    }
    .nav-item {
        color: white; text-decoration: none; font-size: 18px; font-weight: 500;
        transition: all 0.3s ease-in-out;
    }
    .nav-item:hover { text-decoration: underline; color: #dce9ff; }
    .active { text-decoration: underline; font-weight: 600; }
    .fade {
        animation: fadeEffect 0.5s; 
    }
    @keyframes fadeEffect {
        from {opacity: 0;} to {opacity: 1;}
    }
    footer {
        text-align: center; padding: 1rem; margin-top: 2rem; 
        color: #555; border-top: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------- NAVBAR -----------------
selected_tab = st.session_state.get("selected_tab", "Home")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🏠 Home"):
        selected_tab = "Home"
with col2:
    if st.button("📊 Analysis"):
        selected_tab = "Analysis"
with col3:
    if st.button("⚙️ Settings"):
        selected_tab = "Settings"

st.session_state.selected_tab = selected_tab

# ----------------- MODEL TRAINING -----------------
@st.cache_data
def train_model():
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/ankurdome/sonar-dataset/main/sonar.csv", header=None)
    except:
        st.warning("Online dataset unavailable. Loading local fallback...")
        df = pd.read_csv("sonar.csv", header=None)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, "sonar_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")

    return model, scaler, le, acc, df

model, scaler, le, acc, df = train_model()

# ----------------- HOME TAB -----------------
if selected_tab == "Home":
    st.markdown("<div class='fade'>", unsafe_allow_html=True)
    st.title("🪨 SONAR: Rock vs Mine")
    st.write("Predict whether the sonar return indicates a **Rock** or a **Mine** based on acoustic features.")
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("rock.jpg"):
            st.image("rock.jpg", caption="Rock", use_container_width=True)
    with col2:
        if os.path.exists("mine.jpg"):
            st.image("mine.jpg", caption="Mine", use_container_width=True)
    
    st.subheader("🔹 Upload Dataset or Test Single Sample")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file, header=None)
        if data.shape[1] == 61:
            st.success("✅ Dataset already labeled.")
            label_counts = data.iloc[:, -1].value_counts()
            st.bar_chart(label_counts)
            st.write(label_counts)
        else:
            scaled = scaler.transform(data)
            preds = model.predict(scaled)
            decoded = le.inverse_transform(preds)
            data["Predicted Label"] = decoded
            st.dataframe(data)
            st.bar_chart(pd.Series(decoded).value_counts())
    else:
        st.write("Or enter a single CSV row below:")
        example = "0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111, 0.1609, 0.1582, 0.2238, 0.0645, 0.0660, 0.2273, 0.3100, 0.2999, 0.5078, 0.4797, 0.5783, 0.5071, 0.4328, 0.5550, 0.6711, 0.6415, 0.7104, 0.8080, 0.6791, 0.3857, 0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943, 0.2744, 0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324, 0.0232, 0.0027, 0.0065, 0.0159, 0.0072, 0.0167, 0.0180, 0.0084, 0.0090, 0.0032"
        sample = st.text_area("Enter 60 comma-separated values:", example)
        if st.button("Predict"):
            try:
                values = np.array([list(map(float, sample.split(",")))])
                scaled = scaler.transform(values)
                pred = model.predict(scaled)
                label = le.inverse_transform(pred)[0]
                st.success(f"✅ Prediction: **{label}**")
            except Exception as e:
                st.error("Invalid input. Please check formatting.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- ANALYSIS TAB -----------------
elif selected_tab == "Analysis":
    st.markdown("<div class='fade'>", unsafe_allow_html=True)
    st.title("📊 Model Analysis")

    y = le.fit_transform(df.iloc[:, -1])
    X = scaler.transform(df.iloc[:, :-1])
    preds = model.predict(X)

    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    st.pyplot(fig)

    st.metric("Model Accuracy", f"{acc*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- SETTINGS TAB -----------------
elif selected_tab == "Settings":
    st.markdown("<div class='fade'>", unsafe_allow_html=True)
    st.title("⚙️ Settings & Environment")
    st.write("**Python Environment:**")
    st.code("Python 3.12 | scikit-learn | pandas | streamlit | numpy")
    st.write("**Model Info:** K-Nearest Neighbors (KNN), distance-weighted, n_neighbors=5")
    st.write(f"**Trained Accuracy:** {acc*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- FOOTER -----------------
st.markdown(
    """
    <footer>
    Developed by <b>Ankur Dome</b> 💻 |
    <a href="https://github.com/ankurdome" target="_blank">GitHub</a> |
    <a href="https://linkedin.com/in/ankurdome" target="_blank">LinkedIn</a>
    </footer>
    """,
    unsafe_allow_html=True
)
