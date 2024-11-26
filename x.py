import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv')
    data = data.drop(columns=['Unnamed: 32'], errors='ignore')
    g=data.drop(columns=['diagnosis'])
    columns_to_use = [
        'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
        'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave points_mean'
    ]
    data = data[columns_to_use]
    return data

# Train the models and return results
@st.cache_data
def train_models(data):
    X = data.drop(columns=['diagnosis'])
    y = LabelEncoder().fit_transform(data['diagnosis'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}
    
    param_grid = {
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2', 'none']}
}

    results = []
    for model_name, model in models.items():
        grid = GridSearchCV(model, param_grid.get(model_name, {}), cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Best Model": best_model
        })

    return results, scaler

# Load and preprocess the data
data = load_data()
results, scaler = train_models(data)

# Streamlit App Layout
st.title("🎗️ Breast Cancer Diagnosis Prediction")
st.markdown(
    """
    This application predicts whether a tumor is **benign** or **malignant** 
    using various machine learning models trained on clinical data.
    """
)

# Layout: Sidebar for input and actions
st.sidebar.header("🔢 Input Features")
radius_mean = st.sidebar.number_input("Radius Mean", min_value=0.0, max_value=100.0, value=0.0)
texture_mean = st.sidebar.number_input("Texture Mean", min_value=0.0, max_value=100.0, value=0.0)
perimeter_mean = st.sidebar.number_input("Perimeter Mean", min_value=0.0, max_value=1000.0, value=0.0)
area_mean = st.sidebar.number_input("Area Mean", min_value=0.0, max_value=10000.0, value=0.0)
smoothness_mean = st.sidebar.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, value=0.0)
compactness_mean = st.sidebar.number_input("Compactness Mean", min_value=0.0, max_value=1.0, value=0.0)
concavity_mean = st.sidebar.number_input("Concavity Mean", min_value=0.0, max_value=1.0, value=0.0)
concave_points_mean = st.sidebar.number_input("Concave Points Mean", min_value=0.0, max_value=1.0, value=0.0)

# Sidebar: Predict button
if st.sidebar.button("Predict"):
    input_data = np.array([[
        radius_mean, texture_mean, perimeter_mean, area_mean,
        smoothness_mean, compactness_mean, concavity_mean, concave_points_mean
    ]])
    input_data_scaled = scaler.transform(input_data)

    # Use the best model for prediction
    best_model = max(results, key=lambda x: x['F1 Score'])['Best Model']
    prediction = best_model.predict(input_data_scaled)

    # Display prediction result
    st.sidebar.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.sidebar.success("Malignant")
    else:
        st.sidebar.success("Benign")

# Main Section: Analysis and Results
st.header("📊 Results and Analysis")

tabs = st.tabs([
    "Dataset Preview", 
    "Feature Histograms", 
    "Boxplots", 
    "Model Scores", 
    "Confusion Matrices", 
    "Best Model"
])

# Tab 1: Dataset Preview
with tabs[0]:
    st.subheader("Dataset Preview")
    st.write(data.head())

# Tab 2: Feature Histograms
with tabs[1]:
    st.subheader("Feature Distributions")
    features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean'
    ]
    num_features = len(features)
    num_cols = 2
    num_rows = (num_features + num_cols - 1) // num_cols
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
    ax = ax.flatten()
    for i, feature in enumerate(features):
        ax[i].hist(data[feature], bins=20, alpha=0.7, color='skyblue')
        ax[i].set_title(feature)
        ax[i].set_xlabel("Value")
        ax[i].set_ylabel("Frequency")
    for j in range(len(features), len(ax)):
        fig.delaxes(ax[j])
    plt.tight_layout()
    st.pyplot(fig)


# Tab 3: Boxplots
with tabs[2]:
    st.subheader("Boxplots for Feature Distribution")
    
    # Add radio button to choose before or after scaling
    scaling_option = st.radio("View Boxplot", ("Before Scaling", "After Scaling"))
    
    # If after scaling, apply scaling to the data
    if scaling_option == "After Scaling":
        # Apply scaling
        scaled_data = pd.DataFrame(scaler.transform(data.drop(columns=['diagnosis'])), columns=data.columns[1:])
        selected_features = st.multiselect("Select Features for Boxplot", scaled_data.columns, default=scaled_data.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=scaled_data[selected_features], ax=ax)
        ax.set_title("Boxplot of Selected Features (After Scaling)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        # Before scaling, use the original data
        selected_features = st.multiselect("Select Features for Boxplot", data.columns[1:], default=data.columns[1:])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=data[selected_features], ax=ax)
        ax.set_title("Boxplot of Selected Features (Before Scaling)")
        plt.xticks(rotation=45)
        st.pyplot(fig)


# Tab 4: Model Scores
with tabs[3]:
    st.subheader("Model Performance Metrics")
    for result in results:
        st.markdown(f"**{result['Model']}**")
        st.write(f"Accuracy: {result['Accuracy']:.2f}")
        st.write(f"F1 Score: {result['F1 Score']:.2f}")
        st.write(f"Precision: {result['Precision']:.2f}")
        st.write(f"Recall: {result['Recall']:.2f}")
        st.write("---")

# Tab 5: Confusion Matrices
with tabs[4]:
    st.subheader("Confusion Matrices")
    for result in results:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(result["Confusion Matrix"], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{result['Model']} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# Tab 6: Best Model Details
with tabs[5]:
    st.subheader("Best Model Details")
    best_result = max(results, key=lambda x: x['F1 Score'])
    st.markdown(f"### **{best_result['Model']}**")
    st.write(f"**Accuracy:** {best_result['Accuracy']:.2f}")
    st.write(f"**F1 Score:** {best_result['F1 Score']:.2f}")
    st.write(f"**Precision:** {best_result['Precision']:.2f}")
    st.write(f"**Recall:** {best_result['Recall']:.2f}")
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(best_result["Confusion Matrix"], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f"{best_result['Model']} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
