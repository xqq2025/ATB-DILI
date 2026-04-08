import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ------------------ Page configuration ------------------
st.set_page_config(page_title="Pediatric Anti-TB DILI Risk Prediction", layout="wide")
st.title("🔬 Pediatric ATB-DILI Risk Prediction (Liver Function Baseline Stratification)")
st.markdown("This tool predicts drug-induced liver injury (DILI) risk in children receiving anti-tuberculosis treatment, using two baseline models (**Normal** and **Abnormal**) based on initial liver function. It provides SHAP-based individual explanations for interpretability.")

# ------------------ Load models, scalers, and feature lists ------------------
@st.cache_resource
def load_models():
    normal_model = joblib.load('normal_baseline_model.pkl')
    normal_scaler = joblib.load('normal_baseline_scaler.pkl')
    normal_features = joblib.load('normal_baseline_features.pkl')
    
    abnormal_model = joblib.load('abnormal_baseline_model.pkl')
    abnormal_scaler = joblib.load('abnormal_baseline_scaler.pkl')
    abnormal_features = joblib.load('abnormal_baseline_features.pkl')
    
    return (normal_model, normal_scaler, normal_features,
            abnormal_model, abnormal_scaler, abnormal_features)

normal_model, normal_scaler, normal_features, abnormal_model, abnormal_scaler, abnormal_features = load_models()

# ------------------ Two-column layout ------------------
col1, col2 = st.columns(2)

with col1:
    st.header("🧪 Normal Baseline Group")
    st.caption("Features: " + ", ".join(normal_features))
    
    # Input fields for normal group (all numerical)
    input_dict_normal = {}
    for feat in normal_features:
        input_dict_normal[feat] = st.number_input(f"{feat}", value=0.0, step=0.1, key=f"norm_{feat}")
    
    if st.button("Predict Risk (Normal)", key="norm_btn"):
        # Prepare input
        input_df = pd.DataFrame([input_dict_normal])
        # Standardize
        input_scaled = normal_scaler.transform(input_df)
        # Predict probability
        prob = normal_model.predict_proba(input_scaled)[0, 1]
        st.metric("DILI Risk Probability", f"{prob:.2%}")
        
        # SHAP explanation
        explainer = shap.LinearExplainer(normal_model, normal_scaler.mean_.reshape(1, -1))
        shap_values = explainer.shap_values(input_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                             base_values=explainer.expected_value,
                                             data=input_scaled[0],
                                             feature_names=normal_features),
                            show=False)
        plt.tight_layout()
        st.pyplot(fig)

with col2:
    st.header("⚠️ Abnormal Baseline Group")
    st.caption("Features: " + ", ".join(abnormal_features))
    
    input_dict_abnormal = {}
    for feat in abnormal_features:
        # ----- Special handling for binary variable -----
        if feat == 'Prophylactic hepatoprotectant use':
            selected = st.selectbox(f"{feat}", options=["No", "Yes"], key=f"abnorm_{feat}")
            input_dict_abnormal[feat] = 1 if selected == "Yes" else 0
        else:
            input_dict_abnormal[feat] = st.number_input(f"{feat}", value=0.0, step=0.1, key=f"abnorm_{feat}")
    
    if st.button("Predict Risk (Abnormal)", key="abnorm_btn"):
        input_df = pd.DataFrame([input_dict_abnormal])
        input_scaled = abnormal_scaler.transform(input_df)
        prob = abnormal_model.predict_proba(input_scaled)[0, 1]
        st.metric("DILI Risk Probability", f"{prob:.2%}")
        
        explainer = shap.LinearExplainer(abnormal_model, abnormal_scaler.mean_.reshape(1, -1))
        shap_values = explainer.shap_values(input_scaled)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                             base_values=explainer.expected_value,
                                             data=input_scaled[0],
                                             feature_names=abnormal_features),
                            show=False)
        plt.tight_layout()
        st.pyplot(fig)