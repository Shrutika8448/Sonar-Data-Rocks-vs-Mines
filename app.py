import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Sonar Rock vs Mine Classifier", page_icon="🎵", layout="wide")
st.title("🎵 Sonar Rock vs Mine Classifier")

# ---- Upload file ----
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

# ---- Function to train model (used if unlabeled data) ----
@st.cache_resource
def train_knn_model():
    data = pd.read_csv("sonarall-data.csv", header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    param_grid = {'n_neighbors': np.arange(1, 20)}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    best_k = grid.best_params_['n_neighbors']

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)

    return model, scaler, le, best_k

# ---- Visualization helper ----
def plot_label_distribution(labels, title):
    label_counts = pd.Series(labels).value_counts().sort_index()
    labels_display = label_counts.index.map({'R': 'Rock', 'M': 'Mine'}).tolist() if 'R' in label_counts.index else label_counts.index.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.write("### 📊 Bar Chart")
        fig, ax = plt.subplots()
        sns.barplot(x=labels_display, y=label_counts.values, palette="coolwarm", ax=ax)
        ax.set_xlabel("Label")
        ax.set_ylabel("Count")
        ax.set_title(title)
        st.pyplot(fig)

    with col2:
        st.write("### 🥧 Pie Chart")
        fig2, ax2 = plt.subplots()
        ax2.pie(label_counts.values, labels=labels_display, autopct="%1.1f%%", colors=sns.color_palette("coolwarm", len(label_counts)))
        ax2.set_title(title)
        st.pyplot(fig2)

# ---- Main logic ----
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)
    st.subheader("📋 Dataset Preview")
    st.dataframe(data.head())

    # Case 1: Labeled dataset
    if data.shape[1] == 61:
        st.success("✅ Labeled dataset detected (contains R/M column).")

        labels = data.iloc[:, -1]
        rock_count = (labels == 'R').sum()
        mine_count = (labels == 'M').sum()

        st.write(f"**Total Samples:** {len(labels)}")
        st.write(f"🪨 Rocks: {rock_count}")
        st.write(f"💣 Mines: {mine_count}")

        plot_label_distribution(labels, "Actual Label Distribution")

    # Case 2: Unlabeled dataset
    elif data.shape[1] == 60:
        st.warning("⚙️ Unlabeled dataset detected — predicting Rock/Mine using trained KNN model...")

        with st.spinner("Training model and predicting..."):
            model, scaler, le, best_k = train_knn_model()
            input_scaled = scaler.transform(data)
            predictions = model.predict(input_scaled)
            predicted_labels = le.inverse_transform(predictions)

        rock_count = np.sum(predicted_labels == 'R')
        mine_count = np.sum(predicted_labels == 'M')

        st.write(f"✅ **Prediction completed! (Best k = {best_k})**")
        st.write(f"🪨 Predicted Rocks: {rock_count}")
        st.write(f"💣 Predicted Mines: {mine_count}")

        plot_label_distribution(predicted_labels, "Predicted Label Distribution")

        st.subheader("🔹 Predicted Labels Preview")
        pred_df = pd.DataFrame({
            "Sample No.": np.arange(1, len(predicted_labels) + 1),
            "Predicted Label": np.where(predicted_labels == 'R', 'Rock', 'Mine')
        })
        st.dataframe(pred_df.head(20))

    else:
        st.error("❌ Invalid file format. Expected 60 or 61 columns.")
else:
    st.info("👆 Please upload a CSV file to begin.")
