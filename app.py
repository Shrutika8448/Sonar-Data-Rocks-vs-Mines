# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from io import StringIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------- Page config ----------------
st.set_page_config(page_title="SONAR: Rock vs Mine", layout="wide", initial_sidebar_state="expanded")

# ---------------- Theme & CSS ----------------
PRIMARY = "#007BFF"  # blue accent
CARD_BG = "#f8fafc"
st.markdown(f"""
<style>
/* layout */
.app-shell {{ padding: 0 12px 40px 12px; }}

/* sidebar custom */
.custom-sidebar {{
  width: 220px;
  padding: 18px;
  background: white;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(2,6,23,0.06);
  margin-bottom: 14px;
}}
.nav-btn {{
  display: block;
  width: 100%;
  text-align: left;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid transparent;
  margin-bottom: 8px;
  font-weight: 600;
  color: #0b1220;
  background: transparent;
  cursor: pointer;
}}
.nav-btn:hover {{ background: #f1f8ff; }}
.nav-active {{
  background: {PRIMARY};
  color: white;
  border: 1px solid rgba(0,0,0,0.04);
}}
.header {{
  padding: 8px 0 18px 0;
  text-align: left;
}}
.h1 {{
  color: #0b1220;
  font-size: 28px;
  font-weight: 800;
  margin: 6px 0 2px 0;
}}
.h-sub {{
  color: #334155;
  margin-bottom: 8px;
}}
.card {{
  background: {CARD_BG};
  padding: 12px;
  border-radius: 10px;
  box-shadow: 0 1px 4px rgba(2,6,23,0.04);
}}
.small-muted {{ color:#64748b; font-size:13px; }}

/* fade animation for page content */
@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(6px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.page-content > * {{
  animation: fadeIn 0.38s ease;
}}

/* footer */
.footer {{
  text-align: center;
  color: #64748b;
  padding-top: 18px;
  padding-bottom: 40px;
  font-size: 14px;
}}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar (custom) ----------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "single_sample_input" not in st.session_state:
    # default example sample
    st.session_state.single_sample_input = ("0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,"
                                             "0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,"
                                             "0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,"
                                             "0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,"
                                             "0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,"
                                             "0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032")

# left column will be manual sidebar
sidebar_col, main_col = st.columns([1.15, 4.85])
with sidebar_col:
    st.markdown('<div class="custom-sidebar">', unsafe_allow_html=True)
    st.markdown('<div class="header"><div class="h1">SONAR</div><div class="h-sub">Rock vs Mine</div></div>', unsafe_allow_html=True)

    # sidebar: training dataset uploader
    st.markdown("**Training data (labeled)**", unsafe_allow_html=True)
    train_u = st.file_uploader("Upload train CSV (61 cols)", type=["csv"], key="train_uploader")
    st.markdown("<div class='small-muted'>If you don't upload, app will try to use sonarall-data.csv in app folder.</div>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)

    # nav buttons
    def nav_click(target):
        st.session_state.page = target

    # show buttons with active class
    def nav_button(label, target):
        cls = "nav-btn nav-active" if st.session_state.page == target else "nav-btn"
        if st.button(label, key=f"nav_{target}"):
            nav_click(target)
        # emulate active look
        if st.session_state.page == target:
            st.markdown(f"<script>const b=document.querySelector('[data-testid=\"stButton\"][aria-label=\"{label}\"]'); if(b){{b.classList.add('nav-active');}}</script>", unsafe_allow_html=True)

    nav_button("Home", "Home")
    nav_button("Analysis", "Analysis")
    nav_button("Settings", "Settings")

    st.markdown("---", unsafe_allow_html=True)
    st.markdown("**Quick actions**", unsafe_allow_html=True)
    if st.button("Load example sample"):
        st.session_state.single_sample_input = st.session_state.single_sample_input  # already set; this is explicit user trigger
    if st.button("Clear sample"):
        st.session_state.single_sample_input = ""

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Helpers & Model ----------------
@st.cache_data
def load_local_default():
    p = "sonarall-data.csv"
    if os.path.exists(p):
        try:
            df = pd.read_csv(p, header=None)
            return df
        except Exception:
            return None
    return None

@st.cache_resource
def train_knn_cached(df, k=3):
    # expects df: 60 features + label column
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
    # try grid for k if user wants later; keep simple now
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(Xtr, ytr)
    ypred = knn.predict(Xte)
    acc = accuracy_score(yte, ypred)
    cm = confusion_matrix(yte, ypred)
    return {"model": knn, "scaler": scaler, "le": le, "acc": acc, "cm": cm, "X_test": Xte, "y_test": yte}

def is_labeled(df):
    if df.shape[1] == 61:
        return True
    last = df.iloc[:, -1]
    if last.dtype == object or last.dtype == 'O':
        vals = last.dropna().unique()
        if any(str(v).upper() in ("R", "M") for v in vals):
            return True
    return False

def plot_bar_pie(labels, title="Distribution"):
    counts = pd.Series(labels).value_counts()
    pretty = ["Rock" if x == "R" else "Mine" if x == "M" else x for x in counts.index]
    # bar
    fig1, ax1 = plt.subplots(figsize=(5,3))
    ax1.bar(pretty, counts.values)
    ax1.set_title(f"{title} — bar")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)
    # pie
    fig2, ax2 = plt.subplots(figsize=(4,3))
    ax2.pie(counts.values, labels=pretty, autopct="%1.1f%%", startangle=90)
    ax2.set_title(f"{title} — pie")
    st.pyplot(fig2)

def plot_pca_2d(X, labels, title="PCA 2D"):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    dfp = pd.DataFrame(X2, columns=["PC1", "PC2"])
    dfp["label"] = labels
    fig, ax = plt.subplots(figsize=(6,4))
    for lab in np.unique(labels):
        subset = dfp[dfp["label"] == lab]
        ax.scatter(subset["PC1"], subset["PC2"], label=("Rock" if lab=="R" else "Mine"))
    ax.legend()
    ax.set_title(title)
    st.pyplot(fig)

# ---------------- Load or build model ----------------
# priority: uploaded train file -> local file -> None
train_df = None
if st.session_state.get("train_uploaded") is None:
    st.session_state["train_uploaded"] = False

if train_u is not None:
    try:
        train_df = pd.read_csv(train_u, header=None)
        st.session_state["train_uploaded"] = True
        st.session_state["train_df"] = train_df
    except Exception:
        train_df = None
else:
    local = load_local_default()
    if local is not None:
        train_df = local
        st.session_state["train_uploaded"] = False
        st.session_state["train_df"] = train_df
    else:
        train_df = st.session_state.get("train_df", None)

# optionally retrain on demand
if "model_obj" not in st.session_state and train_df is not None:
    try:
        st.session_state["model_obj"] = train_knn_cached(train_df, k=3)
    except Exception:
        st.session_state["model_obj"] = None

model_obj = st.session_state.get("model_obj", None)

# ---------------- Page rendering helpers ----------------
def render_home():
    # wrap content so CSS fade applies on children
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.subheader("Welcome — SONAR: Rock vs Mine")
    st.write("Upload a labeled training dataset (sidebar), analyze datasets, or paste a single 60-value sample to predict Rock or Mine.")

    # side-by-side images + quick description
    cols = st.columns([1,1])
    with cols[0]:
        st.image("https://images.unsplash.com/photo-1602524205483-16b6c70e7a5a?auto=format&fit=crop&w=900&q=80",
                 caption="Rock example", use_column_width=True)
    with cols[1]:
        st.image("https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=900&q=80",
                 caption="Underwater scene (illustrative)", use_column_width=True)

    st.markdown("---")
    st.subheader("Upload dataset for analysis")
    uploaded = st.file_uploader("Upload CSV (60 features [+ optional label])", type=["csv"], key="home_upload")
    if uploaded:
        df = pd.read_csv(uploaded, header=None)
        st.write("Preview:")
        st.dataframe(df.head())
        if is_labeled(df):
            st.success("Labeled dataset detected.")
            labels = df.iloc[:, -1].values
            st.write(f"Total: {len(labels)} | Rocks: {(labels=='R').sum()} | Mines: {(labels=='M').sum()}")
            plot_bar_pie(labels, "Actual distribution (uploaded)")
        else:
            st.info("Unlabeled dataset detected.")
            if model_obj is None:
                st.warning("No trained model available. Upload a labeled training dataset in the sidebar.")
            else:
                scaler = model_obj["scaler"]
                le = model_obj["le"]
                model = model_obj["model"]
                X_new = pd.read_csv(uploaded, header=None).values
                Xs = scaler.transform(X_new)
                preds = model.predict(Xs)
                labs = le.inverse_transform(preds)
                st.write(f"Predicted — Total: {len(labs)} | Rocks: {(labs=='R').sum()} | Mines: {(labs=='M').sum()}")
                plot_bar_pie(labs, "Predicted distribution (uploaded)")
                out = pd.DataFrame(X_new)
                out["predicted"] = np.where(labs=="R","Rock","Mine")
                st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode("utf-8"), "preds.csv", "text/csv")

    st.markdown("---")
    st.subheader("Single sample prediction")
    st.write("Paste 60 comma-separated numeric values (0..1). Use the quick buttons in the sidebar to load an example.")
    sample = st.text_area("Single sample (60 values)", value=st.session_state.single_sample_input, height=120, key="single_sample_area")
    if st.button("Predict sample", key="predict_single"):
        txt = sample.strip()
        try:
            vals = [float(x.strip()) for x in txt.split(",") if x.strip()!=""]
            if len(vals) != 60:
                st.error(f"Found {len(vals)} values. Please provide exactly 60 values.")
            else:
                if model_obj is None:
                    st.warning("No trained model available. Upload a labeled training dataset in the sidebar first.")
                else:
                    scaler = model_obj["scaler"]
                    model = model_obj["model"]
                    le = model_obj["le"]
                    Xs = scaler.transform([vals])
                    pred = model.predict(Xs)
                    lab = le.inverse_transform(pred)[0]
                    pretty = "Rock 🪨" if lab=="R" else "Mine 💣"
                    st.success(f"Prediction: **{pretty}**")
                    # show matching image
                    if lab == "R":
                        st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=900&q=80",
                                 caption="Rock example", use_column_width=True)
                    else:
                        st.image("https://images.unsplash.com/photo-1581066312925-7f87c2b9d635?auto=format&fit=crop&w=900&q=80",
                                 caption="Mine-like (illustrative)", use_column_width=True)
        except Exception as e:
            st.error("Could not parse input. Ensure you pasted 60 comma-separated numeric values.")
    st.markdown('</div>', unsafe_allow_html=True)

def render_analysis():
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.header("Analysis")
    st.write("Upload a dataset (labeled or unlabeled) to visualize distributions and (if available) view model metrics.")
    uploaded = st.file_uploader("Upload dataset for analysis (CSV)", type=["csv"], key="analysis_upload")
    if uploaded:
        df = pd.read_csv(uploaded, header=None)
        st.write("Preview:")
        st.dataframe(df.head())
        if is_labeled(df):
            labels = df.iloc[:, -1].values
            st.write(f"Total: {len(labels)} | Rocks: {(labels=='R').sum()} | Mines: {(labels=='M').sum()}")
            plot_bar_pie(labels, "Actual distribution (uploaded)")
            # feature means by label (first 6)
            feat = df.iloc[:, :-1]
            feat["label"] = df.iloc[:, -1]
            st.subheader("Feature means (first 6 features)")
            st.dataframe(feat.groupby("label").mean().iloc[:, :6].T)
            # PCA
            try:
                plot_pca_2d(feat.iloc[:, :-1].values, labels, "PCA (uploaded)")
            except Exception:
                st.info("PCA failed (maybe small dataset).")
        else:
            st.info("Unlabeled dataset: predictions available if a trained model exists.")
            if model_obj is None:
                st.warning("No trained model available.")
            else:
                Xnew = df.values
                scaler = model_obj["scaler"]
                model = model_obj["model"]
                le = model_obj["le"]
                Xs = scaler.transform(Xnew)
                preds = model.predict(Xs)
                labs = le.inverse_transform(preds)
                st.write(f"Predicted — Total: {len(labs)} | Rocks: {(labs=='R').sum()} | Mines: {(labs=='M').sum()}")
                plot_bar_pie(labs, "Predicted distribution (uploaded)")
                try:
                    plot_pca_2d(Xnew, labs, "PCA (predicted labels)")
                except Exception:
                    pass
    else:
        st.info("Upload a dataset to analyze here.")

    st.markdown("---")
    st.subheader("Model performance (if trained)")
    if model_obj is None:
        st.warning("Model not trained. Upload training data in the sidebar to enable performance metrics.")
    else:
        st.write(f"Accuracy (hold-out): {model_obj['acc']*100:.2f}%")
        cm = model_obj["cm"]
        fig, ax = plt.subplots(figsize=(4,3))
        ax.imshow(cm, interpolation='nearest')
        ax.set_title("Confusion matrix")
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")
        st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

def render_settings():
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.header("Settings & Environment")
    st.subheader("Model")
    if train_df is not None:
        st.write(f"- Training data loaded: {train_df.shape[0]} samples, {train_df.shape[1]} columns")
    else:
        st.write("- No training dataset loaded.")
    if model_obj is not None:
        st.write(f"- KNN (k=3, distance-weighted)")
        st.write(f"- Hold-out accuracy: {model_obj['acc']*100:.2f}%")
    else:
        st.write("- Model not trained in this session.")
    st.subheader("Environment")
    st.write("- Python >=3.8, Streamlit, scikit-learn, pandas, numpy, matplotlib")
    st.write("- Tip: include `sonarall-data.csv` in app folder to auto-load training data on deploy.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Render correct page ----------------
page = st.session_state.page
with main_col:
    if page == "Home":
        render_home()
    elif page == "Analysis":
        render_analysis()
    elif page == "Settings":
        render_settings()
    else:
        render_home()

# ---------------- Footer ----------------
st.markdown(f"""<div class="footer">
    Developed by Ankur Dome • <a href="https://github.com/" target="_blank">GitHub</a> • <a href="https://www.linkedin.com/" target="_blank">LinkedIn</a>
</div>""", unsafe_allow_html=True)
