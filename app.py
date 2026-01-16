import streamlit as st
import pandas as pd
import joblib

# --- 1. Load Model & Assets ---
# Use cache to prevent reloading the model on every user interaction
@st.cache_resource
def load_assets():
    model = joblib.load('best_model_rf.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

model, scaler, model_columns = load_assets()

# --- 2. Title & Description ---
st.title("💰 Data Science Salary Predictor")
st.write("This app uses Machine Learning (Random Forest) to predict whether a salary falls into the **Low**, **Medium**, or **High** category.")

# --- 3. User Input Form ---
# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Profile")
    working_year = st.selectbox("Working Year", [2020, 2021, 2022, 2023])
    designation = st.selectbox("Job Title", [
        "Data Scientist", "Data Engineer", "Data Analyst", 
        "Machine Learning Engineer", "Research Scientist", 
        "AI Scientist", "Big Data Engineer"
    ])
    experience = st.selectbox("Experience Level", ["EN", "MI", "SE", "EX"], 
                              format_func=lambda x: {"EN":"Entry Level", "MI":"Mid Level", "SE":"Senior", "EX":"Executive"}[x])
    employment_status = st.selectbox("Employment Status", ["FT", "PT", "CT", "FL"],
                                     format_func=lambda x: {"FT":"Full Time", "PT":"Part Time", "CT":"Contract", "FL":"Freelance"}[x])

with col2:
    st.subheader("Company Details")
    company_size = st.selectbox("Company Size", ["S", "M", "L"],
                                format_func=lambda x: {"S":"Small (<50)", "M":"Medium (50-250)", "L":"Large (>250)"}[x])
    remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100, step=50)
    
    # We use major countries for the demo to keep the list manageable
    company_location = st.selectbox("Company Location", ["US", "DE", "GB", "IN", "FR", "CA", "ES"])
    employee_location = st.selectbox("Employee Location", ["US", "DE", "GB", "IN", "FR", "CA", "ES"])

# --- 4. Prediction Logic ---
if st.button("Predict Salary Category"):
    
    # A. Convert user input into a DataFrame
    input_data = pd.DataFrame({
        'Working_Year': [working_year],
        'Designation': [designation],
        'Experience': [experience],
        'Employment_Status': [employment_status],
        'Employee_Location': [employee_location],
        'Company_Location': [company_location],
        'Company_Size': [company_size],
        'Remote_Working_Ratio': [remote_ratio]
    })
    
    # B. Perform One-Hot Encoding (Same process as in the Notebook)
    input_encoded = pd.get_dummies(input_data)
    
    # C. Column Alignment (CRITICAL STEP)
    # Ensure the input has exactly the same columns as the trained model
    # Missing columns (e.g., unselected categories) are filled with 0
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # D. Scaling (Optional based on training)
    # Since Random Forest is generally robust to scaling, we skip it here unless 
    # the specific saved model was trained on scaled data.
    # input_encoded = scaler.transform(input_encoded)
    
    # E. Prediction
    prediction = model.predict(input_encoded)[0]
    
    # F. Display Result
    st.success(f"Estimated Salary Category: **{prediction}**")
    
    if prediction == 'High':
        st.balloons()