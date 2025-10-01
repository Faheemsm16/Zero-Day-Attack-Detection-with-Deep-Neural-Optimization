# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from fcmeans import FCM
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Network Anomaly Detection", layout="wide")
st.title("⚡ Network Anomaly Detection Dashboard")

# -----------------------------
# Step 1: Choose Input Option
# -----------------------------
input_option = st.radio("Select Input Option:", 
                        ("Upload CSV for Full Training", "Add Single Row for Analysis"))

# Define features
features = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt',
    'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt',
    'synack', 'ackdat', 'smean', 'dmean', 'ct_srv_src', 'ct_state_ttl',
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
    'is_sm_ips_ports'
]
categorical_features = ['proto', 'service', 'state']

# -----------------------------
# Option 1: Upload CSV
# -----------------------------
if input_option == "Upload CSV for Full Training":
    uploaded_file = st.file_uploader("Upload UNSW-NB15 CSV", type="csv")
    
    # fallback: check local GitHub folder
    if uploaded_file is None:
        default_path = os.path.join(os.getcwd(), "UNSW_NB15_training-set.csv")
        if os.path.exists(default_path):
            uploaded_file = default_path
            st.info(f"Using default dataset from GitHub folder: {default_path}")
        else:
            st.warning("Upload a CSV or place UNSW_NB15_training-set.csv in the app folder.")
    
    if uploaded_file:
        if isinstance(uploaded_file, str):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        st.success("Dataset loaded successfully!")
        df.columns = df.columns.str.strip().str.lower()
        available_features = [col for col in features if col in df.columns]
        
        if 'label' not in df.columns:
            st.error("❌ 'label' column missing!")
        else:
            df = df[available_features + ['label']].copy()
            
            # Encode categorical features
            for col in categorical_features:
                if col in df.columns:
                    df[col] = LabelEncoder().fit_transform(df[col])
            
            df['label'] = df['label'].astype(int)
            scaler = StandardScaler()
            df[available_features] = scaler.fit_transform(df[available_features])
            
            X = df[available_features].values
            y = df['label'].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
            
            st.write("✅ Preprocessing Done")
            st.write("Train class distribution:", pd.Series(y_train).value_counts().to_dict())
            st.write("Test class distribution:", pd.Series(y_test).value_counts().to_dict())
            
            # Fuzzy C-Means Clustering
            st.subheader("Fuzzy C-Means Clustering")
            fcm = FCM(n_clusters=2, max_iter=150, m=2.0)
            fcm.fit(X)
            cluster_labels = fcm.predict(X)
            df_clustered = pd.DataFrame(X, columns=available_features)
            df_clustered['Cluster'] = cluster_labels
            df_clustered['Label'] = y
            st.write("FCM clustering done. Sample clusters:")
            st.dataframe(df_clustered.head())
            
            # Train Deep ANN
            st.subheader("Deep ANN Training")
            model_dann = tf.keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])
            model_dann.compile(
                optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
            history = model_dann.fit(X_train, y_train, epochs=50, batch_size=64,
                                     validation_data=(X_test, y_test),
                                     callbacks=[early_stopping], verbose=0)
            st.success("DANN training complete!")
            
            y_pred_dann = (model_dann.predict(X_test) > 0.5).astype("int32")
            
            # Train SVM
            svm_model = SVC(kernel='rbf', random_state=42)
            svm_model.fit(X_train, y_train)
            y_pred_svm = svm_model.predict(X_test)
            
            # Metrics
            st.subheader("Model Performance Comparison")
            def compute_metrics(y_true, y_pred):
                acc = accuracy_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)
                TN, FP, FN, TP = cm.ravel()
                fpr = FP / (FP + TN)
                tpr = TP / (TP + FN)
                return acc, recall, precision, fpr, tpr, cm
            
            dann_metrics = compute_metrics(y_test, y_pred_dann)
            svm_metrics = compute_metrics(y_test, y_pred_svm)
            
            metrics_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall (TPR)", "False Positive Rate"],
                "DANN": [dann_metrics[0], dann_metrics[2], dann_metrics[1], dann_metrics[3]],
                "SVM": [svm_metrics[0], svm_metrics[2], svm_metrics[1], svm_metrics[3]]
            })
            st.table(metrics_df)
            
            # Confusion Matrices
            st.subheader("Confusion Matrices")
            fig, axes = plt.subplots(1, 2, figsize=(12,5))
            sns.heatmap(dann_metrics[5], annot=True, fmt="d", cmap="Blues", ax=axes[0],
                        xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
            axes[0].set_title("DANN")
            sns.heatmap(svm_metrics[5], annot=True, fmt="d", cmap="Oranges", ax=axes[1],
                        xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
            axes[1].set_title("SVM")
            st.pyplot(fig)

# -----------------------------
# Option 2: Add Single Row
# -----------------------------
elif input_option == "Add Single Row for Analysis":
    st.subheader("Input Features Manually")
    input_data = {}
    for col in features:
        if col in categorical_features:
            input_data[col] = st.selectbox(col, options=[0,1,2], index=0)
        else:
            input_data[col] = st.number_input(col, value=0.0)
    
    if st.button("Analyze Single Row"):
        single_df = pd.DataFrame([input_data])
        # Load training dataset from GitHub folder
        default_path = os.path.join(os.getcwd(), "UNSW_NB15_training-set.csv")
        if not os.path.exists(default_path):
            st.error("UNSW_NB15_training-set.csv not found in app folder for training!")
        else:
            df_train = pd.read_csv(default_path)
            df_train.columns = df_train.columns.str.strip().str.lower()
            available_features = [col for col in features if col in df_train.columns]
            
            # Encode categorical
            for col in categorical_features:
                if col in df_train.columns:
                    le = LabelEncoder().fit(df_train[col])
                    df_train[col] = le.transform(df_train[col])
                    if col in single_df.columns:
                        single_df[col] = le.transform(single_df[col])
            
            # Scale
            scaler = StandardScaler()
            df_train[available_features] = scaler.fit_transform(df_train[available_features])
            X_train = df_train[available_features].values
            y_train = df_train['label'].astype(int).values
            
            # Train DANN
            model_dann = tf.keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])
            model_dann.compile(optimizer=tf.keras.optimizers.AdamW(0.001),
                               loss='binary_crossentropy', metrics=['accuracy'])
            model_dann.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            
            # Prediction
            X_input = scaler.transform(single_df[available_features])
            pred = model_dann.predict(X_input)
            st.write(f"Predicted Anomaly Probability: {pred[0][0]:.4f}")
            
            # -----------------------------
            # SHAP for Single Row
            # -----------------------------
            st.subheader("Feature Contribution (SHAP)")
            explainer = shap.Explainer(model_dann, X_train)
            shap_values = explainer(X_input)
            plt.figure(figsize=(10,4))
            shap.waterfall_plot(shap_values[0], max_display=15)
            st.pyplot(plt.gcf())
            plt.clf()
