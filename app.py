import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
import io

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

# ================= Page & theme =================
st.set_page_config(page_title="SONAR Dashboard", page_icon="📊", layout="wide")

PALETTES = {
    "Blue":   {"primary": "#4EA1FF", "bg": "#0E1117", "bg2": "#1B1F24", "text": "#FAFAFA", "accent": "#2C7FB8"},
    "Teal":   {"primary": "#2DD4BF", "bg": "#0B1416", "bg2": "#112023", "text": "#F1F5F9", "accent": "#14B8A6"},
    "Purple": {"primary": "#A78BFA", "bg": "#0F0B16", "bg2": "#1B1426", "text": "#FAF5FF", "accent": "#7C3AED"},
    "Orange": {"primary": "#F59E0B", "bg": "#0F0E0B", "bg2": "#1C1A14", "text": "#FFF7ED", "accent": "#D97706"},
    "Emerald":{"primary": "#10B981", "bg": "#0E1117", "bg2": "#151A22", "text": "#F8FAFC", "accent": "#34D399"},
}

with st.sidebar:
    st.header("🎛️ Controls")
    palette_name = st.selectbox("Palette", list(PALETTES.keys()), index=0)
    dark_mode = st.toggle("Dark background", value=True)
    show_rail = st.toggle("Show icon rail", value=True)
    st.markdown("---")
    st.caption("Celebration")
    celebrate_mine = st.checkbox("Balloons on Mine (single sample)", value=True)
    st.session_state["celebrate_mine"] = celebrate_mine
    st.markdown("---")
    st.caption("Evaluation")
    eval_threshold = st.slider("Probability threshold for 'M'", 0.05, 0.95, value=0.50, step=0.01)
    st.session_state["eval_threshold"] = eval_threshold
    st.markdown("---")
    hide_chrome = st.checkbox("Hide Streamlit menu/footer", value=False)
    st.session_state["hide_chrome"] = hide_chrome

palette = PALETTES[palette_name]

def inject_css(pal, dark=True, rail=True):
    base_bg = pal["bg"] if dark else "#FFFFFF"
    base_text = pal["text"] if dark else "#0F172A"
    secondary_bg = pal["bg2"] if dark else "#F5F7FB"
    rail_css = f"""
    .nav-rail {{
        position: fixed; top: 70px; left: 10px; width: 48px; padding: 8px 6px;
        background: {secondary_bg}; border-radius: 10px; z-index: 999;
        border: 1px solid rgba(255,255,255,0.06);
    }}
    .nav-rail a {{
        display: block; text-align: center; margin: 8px 0; text-decoration: none; color: {base_text};
        font-size: 18px; opacity: 0.85;
    }}
    .nav-rail a:hover {{ opacity: 1; transform: scale(1.05); }}
    """ if rail else ""
    css = f"""
    <style>
    .stApp {{
        background: linear-gradient(160deg, {base_bg} 0%, {secondary_bg} 100%) !important;
        color: {base_text} !important;
    }}
    [data-testid="stSidebar"] {{ background-color: {secondary_bg} !important; color: {base_text} !important; }}
    .kpi {{
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px; padding: 12px 16px; margin-bottom: 10px;
    }}
    .kpi-title {{ font-size: 12px; opacity: 0.8; margin-bottom: 2px; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; color: {pal["primary"]}; }}
    .kpi-sub {{ font-size: 12px; opacity: 0.8; }}
    .card {{
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 14px 16px; margin-bottom: 14px;
    }}
    hr.accent {{ border: none; height: 2px; background: linear-gradient(90deg, transparent, {pal["primary"]}, transparent); }}
    {rail_css}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def inject_hide_chrome(hide: bool):
    if not hide: return
    st.markdown("""
    <style>
    [data-testid="stToolbar"] { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
    header { visibility: hidden !important; }
    </style>
    """, unsafe_allow_html=True)

inject_css(palette, dark=dark_mode, rail=show_rail)
inject_hide_chrome(st.session_state["hide_chrome"])

if show_rail:
    st.markdown("""
    <div class="nav-rail">
      <a href="#overview" title="Overview">🏠</a>
      <a href="#upload--predict" title="Upload">⬆️</a>
      <a href="#charts" title="Charts">📈</a>
      <a href="#evaluation" title="Evaluation">📊</a>
      <a href="#tools" title="Tools">🧰</a>
    </div>
    """, unsafe_allow_html=True)

# ================= Artifacts =================
@st.cache_resource
def load_artifacts():
    model = joblib.load("sonar_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

model, le = load_artifacts()

# ================= Helpers =================
ZERO_WIDTH_PATTERN = re.compile(r'[\u200B-\u200D\uFEFF]')
def strip_zero_width(s: str) -> str: return ZERO_WIDTH_PATTERN.sub("", s).strip()
def clean_df_zero_width(df: pd.DataFrame) -> pd.DataFrame:
    return df.applymap(lambda x: strip_zero_width(x) if isinstance(x, str) else x)

def prepare_features(df_raw: pd.DataFrame):
    df_raw = clean_df_zero_width(df_raw)
    y_true_str = None
    if df_raw.shape[1] >= 61 and df_raw.iloc[:, -1].dtype == object:
        y_true_str = df_raw.iloc[:, -1].astype(str).values
    df_num = df_raw.apply(pd.to_numeric, errors="coerce")
    if df_num.shape[1] == 61 and df_num.iloc[:, -1].isna().all(): df_num = df_num.iloc[:, :-1]
    if df_num.shape[1] > 60: df_num = df_num.iloc[:, :60]
    y_true = None
    if y_true_str is not None and df_num.shape[1] == 60:
        try: y_true = le.transform(y_true_str)
        except Exception: y_true = None
    return df_num, y_true, y_true_str

def predict_array(arr):
    X = np.array(arr, dtype=float).reshape(1, -1)
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        try: proba = model.predict_proba(X)[0]
        except Exception: proba = None
    label = le.inverse_transform([pred])[0]
    return label, proba

def compute_scores(X):
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.shape[1] == 2: return proba[:, 1]
        except Exception: return None
    if hasattr(model, "decision_function"):
        try:
            dec = model.decision_function(X)
            return dec if dec.ndim == 1 else dec[:, 1]
        except Exception: return None
    return None

def kpi(title, value, sub=""):
    st.markdown(f"<div class='kpi'><div class='kpi-title'>{title}</div><div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)

def csv_bytes(df: pd.DataFrame):
    buf = io.StringIO(); df.to_csv(buf, index=False, header=False); return buf.getvalue().encode("utf-8")

# ================= State =================
for k, v in {"df_num": None, "y_true": None, "y_true_str": None, "preds": None, "scores": None}.items():
    if k not in st.session_state: st.session_state[k] = v

# ================= Top bar =================
st.markdown("<h2 id='overview'>SONAR Dashboard</h2>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
rows = 0 if st.session_state.df_num is None else len(st.session_state.df_num)
pred_mines = int((st.session_state.preds == 1).sum()) if st.session_state.preds is not None else 0
pct_mines = f"{(100*pred_mines/rows):.1f}%" if rows else "0.0%"
acc = "—"
if st.session_state.y_true is not None and st.session_state.preds is not None and len(st.session_state.y_true)==len(st.session_state.preds):
    acc = f"{accuracy_score(st.session_state.y_true, st.session_state.preds)*100:.1f}%"
with col1: kpi("Samples loaded", f"{rows}", "Features: 60")
with col2: kpi("Predicted Mines", f"{pred_mines}", f"{pct_mines} of loaded")
with col3: kpi("Accuracy (if labeled)", acc, "vs. uploaded labels")
with col4: kpi("Eval threshold", f"{int(st.session_state['eval_threshold']*100)}%", "for class 'M'")
st.markdown("<hr class='accent'>", unsafe_allow_html=True)

# ================= Upload & Predict =================
st.markdown("<h3 id='upload--predict'>Upload & Predict</h3>", unsafe_allow_html=True)
box_l, box_r = st.columns([2, 1])
with box_l:
    with st.container(border=True):
        uploaded = st.file_uploader("Upload CSV (60 features, or 60+label R/M)", type=["csv"])
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded, header=None)
                df_num, y_true, y_true_str = prepare_features(raw)
                if df_num.shape[1] != 60:
                    st.error(f"Expected 60 feature columns after cleaning, found {df_num.shape[1]}.")
                elif df_num.isna().any().any() or np.isinf(df_num.to_numpy()).any():
                    bad_rows = df_num.index[df_num.isna().any(axis=1)].tolist()
                    st.error(f"Non-numeric or missing values in rows: {bad_rows}.")
                else:
                    st.session_state.df_num = df_num
                    st.session_state.y_true = y_true
                    st.session_state.y_true_str = y_true_str
                    X = df_num.values
                    st.session_state.preds = model.predict(X)
                    st.session_state.scores = compute_scores(X)
                    st.success(f"Loaded {len(df_num)} samples.")
                    st.dataframe(pd.DataFrame({"prediction": le.inverse_transform(st.session_state.preds)}), use_container_width=True)
            except Exception as e:
                st.error(f"Error processing file: {e}")
        cdl, cdr = st.columns(2)
        with cdl:
            if st.session_state.df_num is not None:
                st.download_button("Download cleaned features", data=csv_bytes(st.session_state.df_num),
                                   file_name="cleaned_features.csv", mime="text/csv")
        with cdr:
            st.download_button("Download template (60 features)",
                               data=csv_bytes(pd.DataFrame(np.zeros((1,60)))),
                               file_name="sonar_features_template.csv", mime="text/csv")
        if st.button("Clear data"):
            st.session_state.df_num = None
            st.session_state.y_true = None
            st.session_state.y_true_str = None
            st.session_state.preds = None
            st.session_state.scores = None
            st.rerun()

with box_r:
    with st.container(border=True):
        st.markdown("**Single sample**")
        vals = st.text_area("Enter 60 comma-separated values", height=120, placeholder="0.02,0.0371,...")
        if st.button("Predict sample"):
            try:
                tokens = [strip_zero_width(x) for x in vals.split(",")]
                arr = [float(t) for t in tokens if t != ""]
                if len(arr) != 60:
                    st.error(f"Expected 60 values, got {len(arr)}.")
                else:
                    label, proba = predict_array(arr)
                    st.markdown(f"Result: **{label}**")
                    if proba is not None and len(proba)==2:
                        classes = le.inverse_transform([0,1])
                        st.write({"probabilities": {classes[0]: float(proba[0]), classes[1]: float(proba[1])}})
                    if label == "M" and st.session_state.get("celebrate_mine", True):
                        st.balloons()
            except Exception as e:
                st.error(f"Error: {e}")

# ================= Charts =================
st.markdown("<h3 id='charts'>Charts</h3>", unsafe_allow_html=True)
if st.session_state.df_num is None:
    st.info("Upload data to enable charts.")
else:
    a, b, c = st.columns([1.2, 1.2, 1])
    with a:
        st.markdown("Feature histogram")
        feat_idx = st.number_input("Feature (0-59)", 0, 59, 0, 1, key="hist_idx")
        fig, ax = plt.subplots(figsize=(4.2,3.2))
        ax.hist(st.session_state.df_num.iloc[:, int(feat_idx)], bins=20, color=palette["accent"], alpha=0.9)
        ax.set_title(f"Feature {int(feat_idx)}")
        st.pyplot(fig, use_container_width=True)
    with b:
        st.markdown("Correlation heatmap (subset)")
        n_heat = st.slider("Columns", 5, 60, 20, 5, key="heat_cols")
        corr = st.session_state.df_num.iloc[:, :n_heat].corr()
        fig, ax = plt.subplots(figsize=(5.2,4.2))
        sns.heatmap(corr, cmap="mako", center=0, ax=ax)
        ax.set_title(f"Correlation (first {n_heat})")
        st.pyplot(fig, use_container_width=True)
    with c:
        st.markdown("PCA (2D) scatter")
        try:
            X_scaled = StandardScaler().fit_transform(st.session_state.df_num.values)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            cmap = st.selectbox("Colormap", ["coolwarm","viridis","plasma"], index=0)
            fig, ax = plt.subplots(figsize=(4.2,3.2))
            ax.scatter(X_pca[:,0], X_pca[:,1], c=X_pca[:,0], cmap=cmap, alpha=0.85, s=26)
            ax.set_title("PCA (2 components)")
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.info(f"PCA unavailable: {e}")

# ================= Evaluation =================
st.markdown("<h3 id='evaluation'>Evaluation (with labels)</h3>", unsafe_allow_html=True)
if st.session_state.df_num is None:
    st.info("Upload labeled data to evaluate.")
elif st.session_state.y_true is None:
    st.info("Labels not detected; upload original CSV with R/M label.")
else:
    scores = st.session_state.scores
    if scores is not None:
        thr = st.session_state["eval_threshold"]
        preds_thr = (scores >= thr).astype(int)
        use_thr = st.toggle("Use custom threshold", value=False)
        preds_eval = preds_thr if use_thr else st.session_state.preds
    else:
        preds_eval = st.session_state.preds
        st.info("Classifier lacks probabilities/decision scores; ROC/thresholding may be limited.")
    e1, e2 = st.columns(2)
    with e1:
        labels = le.inverse_transform([0,1])
        cm = confusion_matrix(st.session_state.y_true, preds_eval, labels=[0,1])
        fig, ax = plt.subplots(figsize=(4.2,3.2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
        st.pyplot(fig, use_container_width=True)
    with e2:
        if scores is not None:
            fpr, tpr, _ = roc_curve(st.session_state.y_true, scores)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(4.2,3.2))
            ax.plot(fpr, tpr, color=palette["primary"], label=f"AUC = {roc_auc:.3f}")
            ax.plot([0,1],[0,1],"k--",alpha=0.5)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc="lower right")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("ROC unavailable for this estimator.")

# ================= Tools =================
st.markdown("<h3 id='tools'>Tools</h3>", unsafe_allow_html=True)
t1, t2, t3 = st.columns([1,1,1])
with t1:
    st.markdown("Template CSV")
    st.download_button("Download features-only template", data=csv_bytes(pd.DataFrame(np.zeros((1,60)))),
                       file_name="sonar_features_template.csv", mime="text/csv")
with t2:
    if st.session_state.df_num is not None:
        st.markdown("Cleaned dataset")
        st.download_button("Download cleaned features", data=csv_bytes(st.session_state.df_num),
                           file_name="cleaned_features.csv", mime="text/csv")
    else:
        st.button("Download cleaned features", disabled=True)
with t3:
    st.markdown("Model info")
    try:
        from sklearn import __version__ as skl_ver
    except Exception:
        skl_ver = "unknown"
    steps = getattr(model, "steps", None)
    final_est = model[-1] if steps is not None else model
    st.write({
        "sklearn_version": skl_ver,
        "pipeline_steps": [name for name, _ in steps] if steps else ["<not a Pipeline>"],
        "estimator": final_est.__class__.__name__
    })
