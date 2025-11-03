import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ---------- Page & basic setup ----------
st.set_page_config(page_title="Sonar Rocks vs Mines", page_icon="🌊", layout="wide")

# Color palettes (material-ish)
PALETTES = {
    "Blue":   {"primary": "#4EA1FF", "bg": "#0E1117", "bg2": "#1B1F24", "text": "#FAFAFA", "accent": "#2C7FB8"},
    "Teal":   {"primary": "#2DD4BF", "bg": "#0B1416", "bg2": "#112023", "text": "#F1F5F9", "accent": "#14B8A6"},
    "Purple": {"primary": "#A78BFA", "bg": "#0F0B16", "bg2": "#1B1426", "text": "#FAF5FF", "accent": "#7C3AED"},
    "Orange": {"primary": "#F59E0B", "bg": "#0F0E0B", "bg2": "#1C1A14", "text": "#FFF7ED", "accent": "#D97706"},
}

# ---------- Sidebar: Theme / Palette ----------
with st.sidebar:
    st.header("🎨 Appearance")
    palette_name = st.selectbox("Color palette", list(PALETTES.keys()), index=0)
    palette = PALETTES[palette_name]
    dark_mode = st.toggle("Use dark background", value=True)

# Inject CSS for palette
def inject_css(pal, dark=True):
    base_bg = pal["bg"] if dark else "#FFFFFF"
    base_text = pal["text"] if dark else "#0F172A"
    secondary_bg = pal["bg2"] if dark else "#F5F7FB"
    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(160deg, {base_bg} 0%, {secondary_bg} 100%) !important;
        color: {base_text} !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {secondary_bg} !important;
        color: {base_text} !important;
    }}
    .metric-card {{
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
    }}
    hr.custom {{
        margin: 0.75rem 0 1rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, {pal["primary"]}, transparent);
    }}
    .section-title {{
        font-weight: 600;
        color: {pal["primary"]};
    }}
    .stButton>button {{
        background: {pal["primary"]} !important;
        color: #0B0B0C !important;
        border: 0 !important;
        border-radius: 8px !important;
    }}
    .accent-text {{ color: {pal["accent"]}; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_css(palette, dark=dark_mode)

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("sonar_model.pkl")   # sklearn Pipeline with scaler + classifier
        le = joblib.load("label_encoder.pkl")    # LabelEncoder for ['R','M']
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()
    return model, le

model, le = load_artifacts()

# ---------- Helpers ----------
ZERO_WIDTH_PATTERN = re.compile(r'[\u200B-\u200D\uFEFF]')

def strip_zero_width(s: str) -> str:
    return ZERO_WIDTH_PATTERN.sub("", s).strip()

def clean_df_zero_width(df: pd.DataFrame) -> pd.DataFrame:
    return df.applymap(lambda x: strip_zero_width(x) if isinstance(x, str) else x)

def prepare_features(df_raw: pd.DataFrame):
    """
    Returns:
      df_num: numeric features DataFrame with exactly 60 columns
      y_true: np.ndarray of encoded labels if present, else None
      y_true_str: np.ndarray of string labels ['R','M'] if present, else None
    """
    # Remove zero-width chars possibly introduced by copy-paste/upload
    df_raw = clean_df_zero_width(df_raw)

    # Preserve possible label column before coercion
    y_true_str = None
    if df_raw.shape[1] >= 61:
        last_col = df_raw.iloc[:, -1]
        if last_col.dtype == object:
            y_true_str = last_col.astype(str).values

    # Coerce to numeric; label becomes NaN
    df_num = df_raw.apply(pd.to_numeric, errors="coerce")

    # If 61 columns and last is all NaN (from R/M), drop it
    if df_num.shape[1] == 61 and df_num.iloc[:, -1].isna().all():
        df_num = df_num.iloc[:, :-1]

    # If more than 60 columns, keep first 60
    if df_num.shape[1] > 60:
        df_num = df_num.iloc[:, :60]

    # Encode labels if we detected R/M
    y_true = None
    if y_true_str is not None and df_num.shape[1] == 60:
        try:
            y_true = le.transform(y_true_str)
        except Exception:
            y_true = None

    return df_num, y_true, y_true_str

def predict_array(arr):
    X = np.array(arr, dtype=float).reshape(1, -1)
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            proba = None
    label = le.inverse_transform([pred])[0]
    return label, proba

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig

def plot_roc(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(fpr, tpr, color=palette["primary"], label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return fig

def compute_scores_for_roc(X):
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            dec = model.decision_function(X)
            if dec.ndim == 1:
                return dec
            elif dec.ndim == 2 and dec.shape[1] == 2:
                return dec[:, 1]
        except Exception:
            pass
    return None

# ---------- Header ----------
st.title("🌊 Sonar Rocks vs Mines")
st.markdown('<hr class="custom">', unsafe_allow_html=True)

# Initialize session storage for cross-tab usage
if "df_num" not in st.session_state:
    st.session_state["df_num"] = None
if "y_true" not in st.session_state:
    st.session_state["y_true"] = None
if "y_true_str" not in st.session_state:
    st.session_state["y_true_str"] = None

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Workflow", "🔎 EDA", "📊 Evaluation", "⚙️ Settings"])

with tab1:
    st.subheader("Upload & Predict")
    c_left, c_right = st.columns([2, 1], vertical_alignment="top")

    with c_left:
        uploaded = st.file_uploader("Upload CSV (60 features, or 60+label R/M)", type=["csv"])
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded, header=None)
                df_num, y_true, y_true_str = prepare_features(raw)

                if df_num.shape[1] != 60:
                    st.error(f"Expected 60 feature columns after cleaning, found {df_num.shape[1]}.")
                elif df_num.isna().any().any() or np.isinf(df_num.to_numpy()).any():
                    bad_rows = df_num.index[df_num.isna().any(axis=1)].tolist()
                    st.error(f"Non-numeric or missing values in rows: {bad_rows}. Please clean your CSV.")
                else:
                    # Persist for other tabs
                    st.session_state["df_num"] = df_num
                    st.session_state["y_true"] = y_true
                    st.session_state["y_true_str"] = y_true_str

                    X = df_num.values
                    preds = model.predict(X)
                    labels_pred = le.inverse_transform(preds)

                    st.markdown("**Predictions**")
                    st.dataframe(pd.DataFrame({"prediction": labels_pred}), use_container_width=True)
                    st.balloons()
            except Exception as e:
                st.error(f"Error processing file: {e}")

        # Optional data clearer
        if st.button("Clear loaded data"):
            st.session_state["df_num"] = None
            st.session_state["y_true"] = None
            st.session_state["y_true_str"] = None
            st.rerun()

    with c_right:
        st.markdown("**Single sample**")
        vals = st.text_area("Enter 60 comma-separated values (e.g., 0.02,0.0371,...)", height=140)
        if st.button("Predict single sample"):
            try:
                tokens = [strip_zero_width(x) for x in vals.split(",")]
                arr = [float(t) for t in tokens if t != ""]
                if len(arr) != 60:
                    st.error(f"Expected 60 values, got {len(arr)}.")
                else:
                    label, proba = predict_array(arr)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"**Result:** <span class='accent-text'>{label}</span>", unsafe_allow_html=True)
                    if proba is not None and len(proba) == 2:
                        classes = le.inverse_transform([0, 1])
                        st.write({"probabilities": {classes[0]: float(proba[0]), classes[1]: float(proba[1])}})
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.subheader("Exploratory Data Analysis")
    st.caption("Upload data in the Workflow tab to enable EDA.")
    df_num = st.session_state.get("df_num", None)
    if df_num is None:
        st.info("No data loaded yet.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='section-title'>Feature histogram</div>", unsafe_allow_html=True)
            feat_idx = st.number_input("Feature index (0-59)", min_value=0, max_value=59, value=0, step=1, key="eda_hist_idx")
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            ax.hist(df_num.iloc[:, int(feat_idx)], bins=20, color=palette["accent"], alpha=0.9)
            ax.set_title(f"Feature {int(feat_idx)} distribution")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            st.pyplot(fig, use_container_width=True)

        with c2:
            st.markdown("<div class='section-title'>Correlation (subset)</div>", unsafe_allow_html=True)
            n_heat = st.slider("Columns for heatmap", min_value=5, max_value=60, value=20, step=5, key="heat_cols")
            corr = df_num.iloc[:, :n_heat].corr()
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            sns.heatmap(corr, cmap="mako", center=0, ax=ax)
            ax.set_title(f"Correlation heatmap (first {n_heat} features)")
            st.pyplot(fig, use_container_width=True)

        with c3:
            st.markdown("<div class='section-title'>PCA (2D) scatter</div>", unsafe_allow_html=True)
            try:
                X_scaled = StandardScaler().fit_transform(df_num.values)
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                color_scheme = st.radio("Color map", ["coolwarm", "viridis", "plasma"], horizontal=True, key="pca_cmap")
                fig, ax = plt.subplots(figsize=(4.2, 3.2))
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=X_pca[:, 0], cmap=color_scheme, alpha=0.85, s=28)
                ax.set_title("PCA (2 components)")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.info(f"PCA plot unavailable: {e}")

with tab3:
    st.subheader("Model Evaluation (when labels present)")
    st.caption("Upload a CSV containing 60 features plus the trailing R/M label in the Workflow tab to enable metrics.")
    df_num = st.session_state.get("df_num", None)
    y_true = st.session_state.get("y_true", None)
    if df_num is None:
        st.info("Upload labeled data in the Workflow tab to evaluate.")
    elif y_true is None:
        st.info("Ground-truth labels not detected; cannot compute metrics.")
    else:
        try:
            X = df_num.values
            preds = model.predict(X)
            labels = le.inverse_transform([0, 1])

            cA, cB = st.columns(2)
            with cA:
                st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
                fig = plot_confusion_matrix(y_true, preds, labels=labels)
                st.pyplot(fig, use_container_width=True)

            with cB:
                st.markdown("<div class='section-title'>ROC Curve</div>", unsafe_allow_html=True)
                scores = compute_scores_for_roc(X)
                if scores is not None:
                    fig = plot_roc(y_true, scores)
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("Classifier does not provide probabilities or decision scores; ROC unavailable.")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

with tab4:
    st.subheader("Settings & Tips")
    st.markdown("- Use the sidebar to switch color palettes and background mode.", unsafe_allow_html=True)
    st.markdown("- Upload the original Sonar CSV (60 features + R/M) for evaluation plots; the label column is auto-dropped for prediction.", unsafe_allow_html=True)
    st.markdown("- Paste single samples in the Workflow tab; invisible characters are auto-removed.", unsafe_allow_html=True)
    st.markdown('<hr class="custom">', unsafe_allow_html=True)
    st.code(
        "pip install streamlit scikit-learn pandas numpy joblib matplotlib seaborn",
        language="bash"
    )
