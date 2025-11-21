# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import os

st.set_page_config(page_title="Rock vs Mine Classifier", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- THEME / CSS ----------
PRIMARY = "#0b69ff"
st.markdown(f"""
<style>
body {{
    background-color: #ffffff;
    color: #0f172a;
}}
.header {{
    text-align: center;
    padding: 6px 0 0 0;
}}
.nav {{
    display: flex;
    gap: 18px;
    justify-content: center;
    margin-bottom: 12px;
}}
.nav button {{
    background: white;
    border: 2px solid {PRIMARY};
    color: {PRIMARY};
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
}}
.card {{
    background: #f8fafc;
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 1px 4px rgba(2,6,23,0.06);
}}
.footer {{
    color: #64748b;
    font-size: 13px;
    text-align: center;
    padding: 18px 0;
}}
.small-muted {{
    color: #64748b;
    font-size: 13px;
}}
</style>
""", unsafe_allow_html=True)

# ---------- NAVBAR ----------
if "page" not in st.session_state:
    st.session_state.page = "Home"

nav_cols = st.columns([1, 1, 1, 6, 1])
with nav_cols[1]:
    if st.button("Home"):
        st.session_state.page = "Home"
with nav_cols[2]:
    if st.button("Analysis"):
        st.session_state.page = "Analysis"
with nav_cols[3]:
    if st.button("Settings"):
        st.session_state.page = "Settings"

# ---------- Helper utils ----------
@st.cache_data
def load_default_training():
    local = "sonar.csv"
    if os.path.exists(local):
        return pd.read_csv(local, header=None)
    return None

# ---------- SVM Training ----------
@st.cache_resource
def train_svm(data: pd.DataFrame, C_value=2):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    svm = SVC(kernel="rbf", C=C_value, gamma="scale")
    svm.fit(Xtr, ytr)

    ypred = svm.predict(Xte)
    acc = accuracy_score(yte, ypred)
    cm = confusion_matrix(yte, ypred)

    return {"model": svm, "scaler": scaler, "le": le, "acc": acc, "cm": cm, "X_test": Xte, "y_test": yte}

def is_labeled(df: pd.DataFrame):
    if df.shape[1] == 61:
        return True
    last = df.iloc[:, -1]
    if last.dtype == object:
        vals = last.dropna().unique()
        if any(str(v).upper() in ("R", "M") for v in vals):
            return True
    return False

def plot_bar_pie(labels, title):
    counts = pd.Series(labels).value_counts()
    idx = counts.index.tolist()
    pretty = ["Rock" if x == "R" else "Mine" if x == "M" else x for x in idx]

    fig1, ax1 = plt.subplots(figsize=(5,3))
    sns.barplot(x=pretty, y=counts.values, ax=ax1)
    ax1.set_title(f"{title} (bar)")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(4,3))
    ax2.pie(counts.values, labels=pretty, autopct="%1.1f%%", startangle=90)
    ax2.set_title(f"{title} (pie)")
    st.pyplot(fig2)

# ---------- Sidebar ----------
st.sidebar.header("Model / Data")
train_upload = st.sidebar.file_uploader("Upload labeled training CSV (61 cols: 60 features + R/M)", type=["csv"])

use_default = False
train_df = None

if train_upload:
    try:
        train_df = pd.read_csv(train_upload, header=None)
        st.sidebar.success("Training file loaded.")
    except:
        st.sidebar.error("Failed to read uploaded file.")
else:
    default_df = load_default_training()
    if default_df is not None:
        train_df = default_df
        use_default = True
        st.sidebar.info("Using local sonar.csv as training dataset (found in app folder).")

# SVM hyperparameter
st.sidebar.markdown("### Model settings")
C_value = st.sidebar.number_input("SVM C Value", min_value=1, max_value=10, value=2, step=1)
retrain = st.sidebar.button("Retrain model (if data present)")

model_obj = None
if train_df is not None:
    if retrain or ("model_obj" not in st.session_state):
        try:
            model_obj = train_svm(train_df, C_value=int(C_value))
            st.session_state.model_obj = model_obj
        except:
            st.sidebar.error("Training failed. Check dataset format.")
    else:
        model_obj = st.session_state.get("model_obj")
else:
    model_obj = None

# ---------- PAGES ----------
def page_home():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Home — Project overview")
    st.write("""
    **Rock vs Mine Classifier (SVM)**  
    Upload a labeled training CSV (60 features + final R/M column) in the sidebar.
    """)

    cols = st.columns([2,3])
    with cols[0]:
        st.subheader("How to use")
        st.markdown("""
        1. Upload labeled CSV for training  
        2. Upload any dataset for analysis  
        3. Use single-sample prediction  
        """)
    with cols[1]:
        st.image("rock.jpg")
        st.image("mine.jpg")
    st.markdown('</div>', unsafe_allow_html=True)
    st.subheader("Upload dataset for analysis")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="home_upload")
    if uploaded:
        df = pd.read_csv(uploaded, header=None)
        st.write(df.head())
        if is_labeled(df):
            labels = df.iloc[:, -1].values
            plot_bar_pie(labels, "Actual label distribution")
        else:
            if model_obj is None:
                st.warning("No trained model.")
            else:
                X = df.values
                Xs = model_obj["scaler"].transform(X)
                preds = model_obj["model"].predict(Xs)
                labs = model_obj["le"].inverse_transform(preds)
                plot_bar_pie(labs, "Predicted distribution")

    st.subheader("Single sample prediction")
    sample = st.text_area("Paste 60 comma-separated values")
    if st.button("Predict single sample"):
        if not sample:
            st.error("Paste values first.")
        else:
            try:
                vals = [float(v.strip()) for v in sample.split(",") if v.strip()]
                if len(vals) != 60:
                    st.error(f"{len(vals)} values found, need 60.")
                else:
                    scaler = model_obj["scaler"]
                    model = model_obj["model"]
                    le = model_obj["le"]
                    xs = scaler.transform([vals])
                    pred = model.predict(xs)
                    lab = le.inverse_transform(pred)[0]
                    pretty = "Rock" if lab == "R" else "Mine"
                    st.success(f"Prediction: **{pretty}**")
                    if lab == "R":
                        st.image("rock.jpg")
                    else:
                        st.image("mine.jpg")
            except:
                st.error("Invalid input.")

def page_analysis():
    st.header("Analysis — Visuals & Metrics")

    uploaded = st.file_uploader("Upload dataset for analysis", type=["csv"], key="analysis_upload")
    if uploaded:
        df = pd.read_csv(uploaded, header=None)
        st.write(df.head())

        if is_labeled(df):
            labs = df.iloc[:, -1].values
            plot_bar_pie(labs, "Actual label distribution")
        else:
            if model_obj is None:
                st.warning("No trained model.")
            else:
                X = df.values
                Xs = model_obj["scaler"].transform(X)
                preds = model_obj["model"].predict(Xs)
                labs = model_obj["le"].inverse_transform(preds)
                plot_bar_pie(labs, "Predicted distribution")

    st.subheader("Model performance")
    if model_obj is None:
        st.warning("No trained model.")
    else:
        st.write(f"SVM (C={C_value}) | Test accuracy = {model_obj['acc']*100:.2f}%")
        cm = model_obj["cm"]
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_title("Confusion Matrix (Test Set)")
        st.pyplot(fig)

def page_settings():
    st.header("Settings")
    st.subheader("Model / Hyperparameters")
    st.write(f"- SVM C value = {int(C_value)}")
    st.write("- Kernel = RBF")
    st.write("- gamma = scale")
    st.subheader("Training data")
    if train_df is not None:
        st.write("Training dataset loaded.")
        st.write(train_df.shape)
    else:
        st.write("No training dataset loaded.")

# ---------- ROUTING ----------
if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Analysis":
    page_analysis()
elif st.session_state.page == "Settings":
    page_settings()
else:
    page_home()

# ---------- FOOTER ----------
st.markdown('<div class="footer">Made with Streamlit • Rock vs Mine Classifier (SVM) • White/Blue Theme</div>', unsafe_allow_html=True)

