# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
import os

st.set_page_config(page_title="Rock vs Mine Classifier", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- THEME / CSS ----------
PRIMARY = "#0b69ff"  # blue accent
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

nav_cols = st.columns([1, 1, 1, 6, 1])  # center navbar
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
    # try local file if exists
    local = "sonar.csv"
    if os.path.exists(local):
        return pd.read_csv(local, header=None)
    return None

@st.cache_resource
def train_knn(data: pd.DataFrame, n_neighbors=3):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn.fit(Xtr, ytr)
    ypred = knn.predict(Xte)
    acc = accuracy_score(yte, ypred)
    cm = confusion_matrix(yte, ypred)
    return {"model": knn, "scaler": scaler, "le": le, "acc": acc, "cm": cm, "X_test": Xte, "y_test": yte}

def is_labeled(df: pd.DataFrame):
    # simple heuristic: 61 columns OR last column contains 'R'/'M' strings
    if df.shape[1] == 61:
        return True
    last = df.iloc[:, -1]
    if last.dtype == object:
        # check if values contain R/M
        vals = last.dropna().unique()
        if any(str(v).upper() in ("R", "M") for v in vals):
            return True
    return False

def plot_bar_pie(labels, title):
    counts = pd.Series(labels).value_counts()
    # normalize label names
    idx = counts.index.tolist()
    pretty = ["Rock" if x == "R" else "Mine" if x == "M" else x for x in idx]
    fig1, ax1 = plt.subplots(figsize=(5,3))
    sns.barplot(x=pretty, y=counts.values, ax=ax1)
    ax1.set_title(f"{title} (bar)")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(4,3))
    ax2.pie(counts.values, labels=pretty, autopct="%1.1f%%", startangle=90)
    ax2.set_title(f"{title} (pie)")
    st.pyplot(fig2)

def plot_pca_2d(X, y_labels, title):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)
    dfp = pd.DataFrame(X2, columns=["PC1","PC2"])
    dfp["label"] = y_labels
    dfp["label_pretty"] = dfp["label"].map(lambda v: "Rock" if v=="R" else ("Mine" if v=="M" else v))
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=dfp, x="PC1", y="PC2", hue="label_pretty", ax=ax, s=40)
    ax.set_title(title)
    st.pyplot(fig)

# ---------- Sidebar: training upload / settings ----------
st.sidebar.header("Model / Data")
train_upload = st.sidebar.file_uploader("Upload labeled training CSV (61 cols: 60 features + R/M)", type=["csv"])
use_default = False
train_df = None
if train_upload:
    try:
        train_df = pd.read_csv(train_upload, header=None)
        st.sidebar.success("Training file loaded.")
    except Exception as e:
        st.sidebar.error("Failed to read uploaded file.")
else:
    default_df = load_default_training()
    if default_df is not None:
        train_df = default_df
        use_default = True
        st.sidebar.info("Using local sonar.csv as training dataset (found in app folder).")
    else:
        st.sidebar.warning("No training file uploaded. Upload to enable prediction on unlabeled data & single sample.")

# model hyperparams
st.sidebar.markdown("### Model settings")
knn_k = st.sidebar.number_input("k (neighbors)", min_value=1, max_value=25, value=3, step=1)
retrain = st.sidebar.button("Retrain model (if data present)")

model_obj = None
if train_df is not None:
    if retrain or ("model_obj" not in st.session_state):
        try:
            model_obj = train_knn(train_df, n_neighbors=int(knn_k))
            st.session_state.model_obj = model_obj
        except Exception as e:
            st.sidebar.error("Training failed. Check dataset format (60 feature columns + 1 label column).")
    else:
        model_obj = st.session_state.get("model_obj", None)
else:
    model_obj = None

# ---------- PAGES ----------
def page_home():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Home — Project overview")
    st.write("""
    **Rock vs Mine Classifier** classifies sonar scans (60 features) into **Rock** or **Mine**.
    Upload a labeled training CSV (60 features + final column R/M) in the sidebar to enable model training.
    """)
    cols = st.columns([2,3])
    with cols[0]:
        st.subheader("How to use")
        st.markdown("""
        1. Upload a labeled training CSV in the **sidebar** (if you have one).  
        2. Use **Upload Dataset** to analyze a full CSV (labeled or unlabeled).  
        3. Use **Single Sample** to paste one sample (60 comma-separated values) and predict.  
        """)
        st.markdown("**Tip:** If you include a `sonar.csv` file in the app folder, the app will auto-use it as default training data.")
    with cols[1]:
        # reliable Unsplash images (add parameters to ensure load)
        st.image("rock.jpg",
                 caption="Rock Image", use_column_width=True)
        st.image("mine.jpg",
                 caption="Mine Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Upload dataset for analysis (labeled/unlabeled)
    st.subheader("Upload dataset for analysis")
    uploaded = st.file_uploader("Upload CSV here (60 features [+ optional label column])", type=["csv"], key="data_upload_home")
    if uploaded:
        df = pd.read_csv(uploaded, header=None)
        st.write("Preview (first 5 rows):")
        st.dataframe(df.head())
        if is_labeled(df):
            st.success("Detected labeled dataset.")
            labels = df.iloc[:, -1].values
            st.write(f"Total samples: {len(labels)} — Rocks: {(labels=='R').sum()} — Mines: {(labels=='M').sum()}")
            plot_bar_pie(labels, "Actual label distribution")
        else:
            st.info("Unlabeled dataset detected. Predictions need trained model.")
            if model_obj is None:
                st.warning("No trained model available. Upload training data in the sidebar to enable predictions.")
            else:
                # predict
                scaler = model_obj["scaler"]
                le = model_obj["le"]
                model = model_obj["model"]
                X_in = df.values
                Xs = scaler.transform(X_in)
                preds = model.predict(Xs)
                labels = le.inverse_transform(preds)
                st.write(f"Predicted — Total: {len(labels)} — Rocks: {(labels=='R').sum()} — Mines: {(labels=='M').sum()}")
                plot_bar_pie(labels, "Predicted label distribution")
                df2 = df.copy()
                df2["Predicted_Label"] = labels
                st.download_button("Download predicted CSV", df2.to_csv(index=False).encode("utf-8"), "predicted.csv", "text/csv")

    st.divider()
    # Single-sample input
    st.subheader("Single sample prediction")
    st.write("Paste 60 comma-separated numeric values (0..1). Example sample available in quick-fill.")
    sample_input = st.text_area("Single sample (60 values)", height=120, key="single_sample_input")
    

    if st.button("Predict single sample"):
        txt = st.session_state.get("single_sample_input", "")
        if not txt:
            st.error("Paste 60 comma-separated values first.")
        else:
            try:
                vals = [float(x.strip()) for x in txt.split(",") if x.strip()!=""]
                if len(vals) != 60:
                    st.error(f"Found {len(vals)} values — please provide exactly 60.")
                else:
                    if model_obj is None:
                        st.warning("No trained model available. Upload training data in the sidebar first.")
                    else:
                        scaler = model_obj["scaler"]
                        model = model_obj["model"]
                        le = model_obj["le"]
                        Xs = scaler.transform([vals])
                        pred = model.predict(Xs)
                        lab = le.inverse_transform(pred)[0]
                        pretty = "Rock" if lab == "R" else "Mine"
                        st.success(f"Prediction: **{pretty}**")
                        # show image
                        if lab == "R":
                            st.image("rock.jpg",
                                     caption="Rock", use_column_width=True)
                        else:
                            st.image("mine.jpg",
                                     caption="Mine", use_column_width=True)
            except Exception as e:
                st.error("Could not parse numbers. Ensure comma-separated floats.")

def page_analysis():
    st.header("Analysis — Visuals & metrics")
    st.write("This page shows dataset-level visualizations and model performance (if training data is available).")
    # Upload dataset to analyze here
    uploaded = st.file_uploader("Upload dataset to analyze (labeled or unlabeled)", type=["csv"], key="upload_analysis")
    if not uploaded:
        st.info("Upload a dataset (csv) to analyze. If you uploaded a training dataset in the sidebar, model metrics are shown below.")
    else:
        df = pd.read_csv(uploaded, header=None)
        st.write("Preview:")
        st.dataframe(df.head())
        if is_labeled(df):
            labels = df.iloc[:, -1].values
            st.write(f"Total: {len(labels)} — Rocks: {(labels=='R').sum()} — Mines: {(labels=='M').sum()}")
            plot_bar_pie(labels, "Actual label distribution (uploaded)")
            # feature stats by label (show means for first 6 features)
            st.subheader("Feature means by label (first 6 features)")
            feat_df = df.iloc[:, :-1].copy()
            feat_df["label"] = df.iloc[:, -1]
            means = feat_df.groupby("label").mean().iloc[:, :6].T
            st.dataframe(means)
            # PCA plot
            plot_pca_2d(feat_df.iloc[:, :].values[:, :60] if feat_df.shape[1]>=61 else feat_df.iloc[:, :-1].values, labels, "PCA (uploaded)")
        else:
            st.info("Unlabeled dataset: predictions available only if a trained model exists.")
            if model_obj is None:
                st.warning("No trained model available. Upload training dataset in the sidebar to enable predicted analysis.")
            else:
                scaler = model_obj["scaler"]
                model = model_obj["model"]
                le = model_obj["le"]
                X = df.values
                Xs = scaler.transform(X)
                preds = model.predict(Xs)
                labs = le.inverse_transform(preds)
                st.write(f"Predicted — Total: {len(labs)} — Rocks: {(labs=='R').sum()} — Mines: {(labs=='M').sum()}")
                plot_bar_pie(labs, "Predicted distribution (uploaded)")
                plot_pca_2d(X, labs, "PCA of uploaded (predicted labels)")

    # show model performance if available
    st.divider()
    st.subheader("Model performance (on hold-out test set)")
    if model_obj is None:
        st.warning("No trained model in session. Upload training dataset in the sidebar and click 'Retrain model'.")
    else:
        st.write(f"Model k = {knn_k} | Test accuracy = {model_obj['acc']*100:.2f}%")
        cm = model_obj["cm"]
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (hold-out)")
        st.pyplot(fig)

def page_settings():
    st.header("Settings & Environment")
    st.subheader("Model / Hyperparameters")
    st.write(f"- K (neighbors): {int(knn_k)}")
    st.write("- KNN weights: distance")
    st.write("- Feature scaling: StandardScaler")

    st.subheader("Files & data")
    if train_df is not None:
        st.write(f"- Training dataset: available ({'default sonar.csv' if use_default else 'uploaded file'})")
        st.write(f"  - samples: {train_df.shape[0]}  features+label cols: {train_df.shape[1]}")
    else:
        st.write("- No training dataset currently loaded.")

    st.subheader("Environment / Dependencies")
    st.write("""
    Python packages used:
    - streamlit
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn
    """)
    st.markdown("**Tip:** Put `sonar.csv` in the app folder to let the app auto-load training data when deployed.")

# ---------- ROUTE ----------
if st.session_state.page == "Home":
    page_home()
elif st.session_state.page == "Analysis":
    page_analysis()
elif st.session_state.page == "Settings":
    page_settings()
else:
    page_home()

# ---------- FOOTER ----------
st.markdown('<div class="footer">Made with Streamlit • Rock vs Mine Classifier • Theme: White + Blue</div>', unsafe_allow_html=True)
