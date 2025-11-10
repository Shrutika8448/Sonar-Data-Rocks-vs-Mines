import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("🔍 Sonar Rock vs Mine Prediction App")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, header=None)
    st.write("### Dataset Preview")
    st.write(data.head())
    st.write("**Shape of dataset:**", data.shape)

    # Split data
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.write("### Model Training and Comparison")

    # ---- KNN ----
    neighbors = np.arange(1, 14)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)

    fig, ax = plt.subplots()
    ax.plot(neighbors, test_accuracy, label='Testing Accuracy')
    ax.plot(neighbors, train_accuracy, label='Training Accuracy')
    ax.set_xlabel('Number of Neighbors')
    ax.set_ylabel('Accuracy')
    ax.set_title('k-NN Varying Number of Neighbors')
    ax.legend()
    st.pyplot(fig)

    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    # ---- Logistic Regression ----
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_logistic = log_model.predict(X_test)

    # ---- PCA + Logistic Regression ----
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    model_pca = LogisticRegression(max_iter=1000)
    model_pca.fit(X_train_pca, y_train)
    y_pred_pca = model_pca.predict(X_test_pca)

    # ---- SVM ----
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    # ---- Display Results ----
    st.subheader("Model Performance Metrics")
    st.write("**kNN Accuracy:**", accuracy_score(y_test, y_pred_knn))
    st.write("**Logistic Regression Accuracy:**", accuracy_score(y_test, y_pred_logistic))
    st.write("**PCA + Logistic Regression Accuracy:**", accuracy_score(y_test, y_pred_pca))
    st.write("**SVM Accuracy:**", accuracy_score(y_test, y_pred_svm))

    # Confusion matrices
    st.write("### Confusion Matrices")

    models = {
        "kNN": y_pred_knn,
        "Logistic Regression": y_pred_logistic,
        "PCA + Logistic Regression": y_pred_pca,
        "SVM": y_pred_svm
    }

    for name, preds in models.items():
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        st.pyplot(fig)
else:
    st.info("👆 Please upload a CSV file to begin.")
