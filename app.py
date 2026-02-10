import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# --- MUST BE FIRST: Page Configuration ---
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="wide")

# --- 0. Custom Class Definition (WAJIB ADA) ---
# Harus sama persis dengan yang di Notebook saat training
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.maps = {}
        self.global_mean = 0
        
    def fit(self, X, y):
        self.global_mean = y.mean()
        for col in self.cols:
            mapper = pd.concat([X[col], y], axis=1).groupby(col).mean().iloc[:, 0]
            self.maps[col] = mapper
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.cols:
            if col in self.maps:
                X_out[col] = X_out[col].map(self.maps[col]).fillna(self.global_mean)
        return X_out

# --- 1. Load Model & Assets ---
@st.cache_resource
def load_assets():
    # Load regressor model (HistGradientBoosting / RandomForest)
    try:
        model = joblib.load('best_model.pkl')
        encoder = joblib.load('encoder.pkl')
        config = joblib.load('config.pkl')
    except Exception as e:
        st.error(f"Error Loading Files: {e}")
        st.warning("Please run the notebook to generate 'best_model.pkl', 'encoder.pkl', and 'config.pkl'.")
        return None, None, None
        
    return model, encoder, config

model, encoder, config = load_assets()

# --- 2. Title & Description ---
st.title("💰 AI Data Science Salary Predictor")
st.markdown("""
This advanced app uses **HistGradientBoosting** to predict the **exact salary value** 
and then categorizes it into **Low, Medium, or High**.
""")

# --- 3. User Input Form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Job Profile")
        designation = st.selectbox("Job Title", [
            "Data Scientist", "Data Engineer", "Data Analyst", 
            "Machine Learning Engineer", "Research Scientist", 
            "AI Scientist", "Big Data Engineer", "Lead Data Scientist",
            "Principal Data Scientist", "Director of Data Science"
        ])
        experience = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"], 
                                format_func=lambda x: {"EN":"Entry Level", "MI":"Mid Level", "SE":"Senior", "EX":"Executive"}[x])
        
    with col2:
        st.subheader("🏢 Company Details")
        company_size = st.selectbox("Company Size", ["S", "M", "L"],
                                    format_func=lambda x: {"S":"Small (<50)", "M":"Medium (50-250)", "L":"Large (>250)"}[x])
        remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100, step=50, help="0=Onsite, 50=Hybrid, 100=Remote")
        
        # Lokasi paling umum (Top 10)
        company_location = st.selectbox("Company Location", ["US", "GB", "CA", "DE", "IN", "FR", "ES", "GR", "JP", "NL", "PT"])
        employee_location = st.selectbox("Employee Location", ["US", "GB", "CA", "DE", "IN", "FR", "ES", "GR", "JP", "NL", "PT"])

    submitted = st.form_submit_button("🚀 Predict Salary")

# --- 4. Prediction Logic ---
if submitted and model is not None:
    
    # A. Prepare Input Data
    input_df = pd.DataFrame([{
        'Designation': designation,
        'Experience': experience,
        'Company_Size': company_size,
        'Remote_Working_Ratio': remote_ratio,
        'Company_Location': company_location,
        'Employee_Location': employee_location
    }])
    
    # B. Feature Engineering (Sama persis dengan Notebook)
    def advanced_feature_engineering_app(df_in):
        df_eng = df_in.copy()
        
        # Size Score
        size_map = {'S': 1, 'M': 2, 'L': 3}
        df_eng['size_score'] = df_eng['Company_Size'].map(size_map).fillna(2)
        
        # Experience Score
        exp_map = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
        df_eng['exp_score'] = df_eng['Experience'].map(exp_map).fillna(2)
        
        # Remote Score
        df_eng['remote_score'] = df_eng['Remote_Working_Ratio']
        
        return df_eng
        
    input_fe = advanced_feature_engineering_app(input_df)
    
    # C. Target Encoding Transform (Pakai encoder dari training)
    cols_to_encode = ['Designation', 'Company_Location', 'Employee_Location']
    feature_cols = ['size_score', 'exp_score', 'remote_score'] + cols_to_encode
    
    # Pastikan urutan kolom sesuai
    input_ready = input_fe[feature_cols].copy()
    
    # Transform
    input_encoded = encoder.transform(input_ready)
    
    # D. Predict Number
    predicted_salary = model.predict(input_encoded)[0]
    
    # E. Convert to Category
    t1 = config['t1']
    t2 = config['t2']
    
    if predicted_salary <= t1:
        category = "Low"
        color = "red"
    elif predicted_salary <= t2:
        category = "Medium"
        color = "orange"
    else:
        category = "High"
        color = "green"

    # --- 5. Display Result ---
    st.divider()
    st.subheader("Prediction Result")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.markdown(f"### Category: :{color}[{category}]")
        st.caption(f"Based on your profile, the salary is in the **{category}** range.")
        
    with col_res2:
        st.metric(label="Estimated Annual Salary (Rupees)", value=f"Rp {predicted_salary:,.0f}")
        
    # Visualisasi Posisi
    st.progress(min(1.0, max(0.0, (predicted_salary / (t2 * 1.5)))))
    st.caption(f"Scale: 0 ... Low (<{t1:,.0f}) ... Medium ... High (>{t2:,.0f}) ...")

    if category == 'High':
        st.balloons()