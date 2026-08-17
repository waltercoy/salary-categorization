import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt
from model_utils import SmoothedTargetEncoder, advanced_feature_engineering, COUNTRY_NAME_MAP, CURRENCY_RATES

# --- 1. Page Configuration & Custom CSS Styling ---
st.set_page_config(
    page_title="Data Science Salary Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium appearance
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .tier-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .tier-low { background-color: #FEE2E2; color: #DC2626; border: 1px solid #FCA5A5; }
    .tier-medium { background-color: #FEF3C7; color: #D97706; border: 1px solid #FCD34D; }
    .tier-high { background-color: #DCFCE7; color: #16A34A; border: 1px solid #86EFAC; }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        height: 2.8rem;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. Load Assets (Model, Encoder, Config, Dataset) ---
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load('best_model.pkl')
        encoder = joblib.load('encoder.pkl')
        config = joblib.load('config.pkl')
        # Validate model execution with dummy sample
        dummy_df = pd.DataFrame([{
            'Designation': 'Data Scientist', 'Experience': 'MI', 'Employment_Status': 'FT',
            'Company_Size': 'M', 'Remote_Working_Ratio': 50, 'Company_Location': 'US',
            'Employee_Location': 'US', 'Working_Year': 2022
        }])
        dummy_fe = advanced_feature_engineering(dummy_df)
        dummy_ready = dummy_fe[config.get('feature_cols', dummy_fe.columns.tolist())]
        dummy_enc = encoder.transform(dummy_ready)
        _ = model.predict(dummy_enc)
        return model, encoder, config
    except Exception as e:
        # Auto-train on-the-fly if pickle version mismatch or missing artifacts on cloud
        try:
            from train import train_and_evaluate
            train_and_evaluate()
            model = joblib.load('best_model.pkl')
            encoder = joblib.load('encoder.pkl')
            config = joblib.load('config.pkl')
            return model, encoder, config
        except Exception as train_err:
            st.error(f"Error initializing model: {train_err}")
            return None, None, None

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("Data_Science_Fields_Salary_Categorization.csv")
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        df['Salary_In_Rupees'] = df['Salary_In_Rupees'].astype(str).str.replace(',', '').astype(float)
        return df
    except Exception:
        return None

model, encoder, config = load_model_assets()
raw_df = load_dataset()

# Populate dynamic dropdown options
if raw_df is not None:
    available_designations = sorted(raw_df['Designation'].dropna().unique().tolist())
    available_company_locs = sorted(raw_df['Company_Location'].dropna().unique().tolist())
    available_employee_locs = sorted(raw_df['Employee_Location'].dropna().unique().tolist())
else:
    available_designations = ["Data Scientist", "Data Engineer", "Data Analyst", "Machine Learning Engineer"]
    available_company_locs = ["US", "GB", "CA", "DE", "IN", "SG"]
    available_employee_locs = ["US", "GB", "CA", "DE", "IN", "SG"]

# Format country labels
def format_country(code):
    name = COUNTRY_NAME_MAP.get(code, code)
    return f"{name} ({code})"

def format_currency_value(amount_inr, curr_code, period="Monthly"):
    curr_info = CURRENCY_RATES.get(curr_code, CURRENCY_RATES['USD'])
    multiplier = (1 / 12.0) if period == "Monthly" else 1.0
    val = amount_inr * curr_info['rate'] * multiplier
    sym = curr_info['symbol']
    suffix = " / month" if period == "Monthly" else " / year"
    return f"{sym} {val:,.0f}{suffix}"


# --- 3. Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/money-bag.png", width=64)
    st.title("Settings")
    
    selected_currency = st.selectbox(
        "💱 Display Currency",
        options=list(CURRENCY_RATES.keys()),
        index=1, # Default to USD
        format_func=lambda x: CURRENCY_RATES[x]['name']
    )
    
    selected_period = st.radio(
        "⏱️ Payment Frequency",
        options=["Monthly", "Yearly"],
        index=0, # Default to Monthly
        format_func=lambda x: "📅 Per Month (Monthly)" if x == "Monthly" else "📆 Per Year (Yearly)"
    )
    
    period_label = "Monthly" if selected_period == "Monthly" else "Annual"
    
    st.divider()
    st.markdown("### 🤖 ML Model Specs")
    if config:
        st.caption(f"**Architecture:** HistGradientBoosting (Log-Target)")
        st.caption(f"**Test Accuracy:** `{config.get('accuracy', 0.7158)*100:.1f}%`")
        st.caption(f"**Weighted F1:** `{config.get('f1_score', 0.7166)*100:.1f}%`")
    
    st.divider()
    st.markdown("Developed by **Aria Firmansyah**")


# --- 4. Main Navigation Tabs ---
st.markdown('<div class="main-title">💼 Data Science Salary Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict global market compensation, benchmark salary tiers, and simulate data career growth.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Salary Predictor", "📈 Market Insights (EDA)", "💡 Career Growth Simulator"])


# ==========================================
# TAB 1: SALARY PREDICTOR
# ==========================================
with tab1:
    st.markdown("### 📝 Enter Professional Profile")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Role & Seniority")
            designation = st.selectbox(
                "Job Title / Designation", 
                options=available_designations, 
                index=available_designations.index("Data Scientist") if "Data Scientist" in available_designations else 0
            )
            
            experience = st.selectbox(
                "Experience Level (Seniority)", 
                options=["EN", "MI", "SE", "EX"],
                index=1,
                format_func=lambda x: {"EN": "🌱 Entry Level / Junior", "MI": "⚡ Mid-Level", "SE": "🚀 Senior Level", "EX": "👑 Executive / Director"}[x]
            )
            
            employment_status = st.selectbox(
                "Employment Type",
                options=["FT", "PT", "CT", "FL"],
                index=0,
                format_func=lambda x: {"FT": "Full-Time", "PT": "Part-Time", "CT": "Contract", "FL": "Freelance"}[x]
            )
            
            working_year = st.selectbox("Market Reporting Year", options=[2022, 2021, 2020], index=0)

        with col2:
            st.subheader("🏢 Company & Location")
            company_size = st.selectbox(
                "Company Size", 
                options=["S", "M", "L"],
                index=1,
                format_func=lambda x: {"S": "Small (< 50 employees)", "M": "Medium (50 - 250 employees)", "L": "Large (> 250 employees)"}[x]
            )
            
            remote_ratio = st.select_slider(
                "Remote Work Policy",
                options=[0, 50, 100],
                value=50,
                format_func=lambda x: {0: "🏢 Onsite (0% Remote)", 50: "🔄 Hybrid (50% Remote)", 100: "🌐 Fully Remote (100%)"}[x]
            )
            
            company_loc = st.selectbox("Company HQ Location", options=available_company_locs, index=available_company_locs.index("US") if "US" in available_company_locs else 0, format_func=format_country)
            employee_loc = st.selectbox("Employee Residence Location", options=available_employee_locs, index=available_employee_locs.index("US") if "US" in available_employee_locs else 0, format_func=format_country)

        submitted = st.form_submit_button("🚀 Calculate Estimated Salary")

    if submitted:
        if model is None or encoder is None or config is None:
            st.error("Model artifacts not found. Please ensure best_model.pkl and encoder.pkl exist.")
        else:
            with st.spinner("Analyzing market profile and running prediction pipeline..."):
                input_df = pd.DataFrame([{
                    'Designation': designation,
                    'Experience': experience,
                    'Employment_Status': employment_status,
                    'Company_Size': company_size,
                    'Remote_Working_Ratio': remote_ratio,
                    'Company_Location': company_loc,
                    'Employee_Location': employee_loc,
                    'Working_Year': working_year
                }])
                
                # Transform features
                input_fe = advanced_feature_engineering(input_df)
                cols_to_encode = config.get('cols_to_encode', ['Designation', 'Company_Location', 'Employee_Location'])
                num_cols = config.get('num_cols', ['size_score', 'exp_score', 'emp_score', 'remote_score', 'is_same_country', 'Working_Year'])
                feature_cols = num_cols + cols_to_encode
                
                input_ready = input_fe[feature_cols].copy()
                input_encoded = encoder.transform(input_ready)
                
                # Predict (Log-scale -> expm1)
                pred_log = model.predict(input_encoded)[0]
                predicted_salary_inr = float(np.expm1(pred_log))
                
                # Determine tier category
                t1 = config['t1']
                t2 = config['t2']
                
                if predicted_salary_inr <= t1:
                    category = "Low"
                    tier_class = "tier-low"
                elif predicted_salary_inr <= t2:
                    category = "Medium"
                    tier_class = "tier-medium"
                else:
                    category = "High"
                    tier_class = "tier-high"

            # --- Display Results ---
            st.divider()
            st.subheader("🎯 Predicted Compensation Result")
            
            res_col1, res_col2 = st.columns([1, 1.4])
            
            with res_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size: 0.9rem; color: #64748B; text-transform: uppercase;">Compensation Tier</span><br>
                    <div style="margin: 12px 0;">
                        <span class="tier-badge {tier_class}">{category} Tier</span>
                    </div>
                    <span style="font-size: 0.85rem; color: #64748B;">Based on global market percentile thresholds</span>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                formatted_primary = format_currency_value(predicted_salary_inr, selected_currency, selected_period)
                st.metric(
                    label=f"Estimated {period_label} Salary ({selected_currency})",
                    value=formatted_primary,
                    help=f"Exact annual benchmark in INR: ₹ {predicted_salary_inr:,.0f} / year"
                )
                
                # Multi-currency overview chips
                c_usd = format_currency_value(predicted_salary_inr, 'USD', selected_period)
                c_myr = format_currency_value(predicted_salary_inr, 'MYR', selected_period)
                c_idr = format_currency_value(predicted_salary_inr, 'IDR', selected_period)
                c_eur = format_currency_value(predicted_salary_inr, 'EUR', selected_period)
                c_inr = format_currency_value(predicted_salary_inr, 'INR', selected_period)
                st.caption(f"**Multi-Currency Overview ({period_label}):** {c_usd} | {c_myr} | {c_idr} | {c_eur} | {c_inr}")

            # Salary Position Bar
            st.markdown("#### 📊 Position in Global Salary Distribution")
            norm_val = min(1.0, max(0.0, predicted_salary_inr / (t2 * 1.5)))
            st.progress(norm_val)
            
            t1_formatted = format_currency_value(t1, selected_currency, selected_period)
            t2_formatted = format_currency_value(t2, selected_currency, selected_period)
            
            col_t1, col_t2, col_t3 = st.columns(3)
            col_t1.caption(f"**🔴 Low Tier:** < {t1_formatted}")
            col_t2.caption(f"**🟡 Medium Tier:** {t1_formatted} – {t2_formatted}")
            col_t3.caption(f"**🟢 High Tier:** > {t2_formatted}")

            if category == "High":
                st.balloons()


# ==========================================
# TAB 2: MARKET INSIGHTS & EDA
# ==========================================
with tab2:
    st.markdown(f"### 📊 Market Compensation Trends ({period_label} View)")
    st.caption("Data-driven exploratory patterns derived directly from the global Data Science salary dataset.")
    
    period_multiplier = (1 / 12.0) if selected_period == "Monthly" else 1.0
    
    if raw_df is not None:
        col_ed1, col_ed2 = st.columns(2)
        
        with col_ed1:
            st.markdown(f"#### 📈 Average Salary by Seniority ({period_label})")
            exp_order = ['EN', 'MI', 'SE', 'EX']
            exp_labels = {'EN': 'Entry Level', 'MI': 'Mid-Level', 'SE': 'Senior', 'EX': 'Executive'}
            
            exp_df = raw_df.groupby('Experience')['Salary_In_Rupees'].mean().reindex(exp_order).reset_index()
            exp_df['Experience_Label'] = exp_df['Experience'].map(exp_labels)
            exp_df['Salary_Converted'] = exp_df['Salary_In_Rupees'] * CURRENCY_RATES[selected_currency]['rate'] * period_multiplier
            
            chart_exp = alt.Chart(exp_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color='#3B82F6').encode(
                x=alt.X('Experience_Label:N', sort=list(exp_labels.values()), title='Experience Level'),
                y=alt.Y('Salary_Converted:Q', title=f'Average Salary ({selected_currency} / {selected_period.lower()})'),
                tooltip=['Experience_Label', alt.Tooltip('Salary_Converted:Q', format=',.0f')]
            ).properties(height=300)
            st.altair_chart(chart_exp, use_container_width=True)

        with col_ed2:
            st.markdown(f"#### 🏢 Average Salary by Company Scale ({period_label})")
            size_order = ['S', 'M', 'L']
            size_labels = {'S': 'Small (<50)', 'M': 'Medium (50-250)', 'L': 'Large (>250)'}
            
            size_df = raw_df.groupby('Company_Size')['Salary_In_Rupees'].mean().reindex(size_order).reset_index()
            size_df['Size_Label'] = size_df['Company_Size'].map(size_labels)
            size_df['Salary_Converted'] = size_df['Salary_In_Rupees'] * CURRENCY_RATES[selected_currency]['rate'] * period_multiplier
            
            chart_size = alt.Chart(size_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color='#10B981').encode(
                x=alt.X('Size_Label:N', sort=list(size_labels.values()), title='Company Scale'),
                y=alt.Y('Salary_Converted:Q', title=f'Average Salary ({selected_currency} / {selected_period.lower()})'),
                tooltip=['Size_Label', alt.Tooltip('Salary_Converted:Q', format=',.0f')]
            ).properties(height=300)
            st.altair_chart(chart_size, use_container_width=True)
            
        st.markdown(f"#### 🏆 Top 10 Highest Paying Roles ({period_label})")
        role_counts = raw_df['Designation'].value_counts()
        valid_roles = role_counts[role_counts >= 3].index
        
        top_roles = raw_df[raw_df['Designation'].isin(valid_roles)].groupby('Designation')['Salary_In_Rupees'].mean().sort_values(ascending=False).head(10).reset_index()
        top_roles['Salary_Converted'] = top_roles['Salary_In_Rupees'] * CURRENCY_RATES[selected_currency]['rate'] * period_multiplier
        
        chart_roles = alt.Chart(top_roles).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color='#6366F1').encode(
            y=alt.Y('Designation:N', sort='-x', title='Role Designation'),
            x=alt.X('Salary_Converted:Q', title=f'Average Salary ({selected_currency} / {selected_period.lower()})'),
            tooltip=['Designation', alt.Tooltip('Salary_Converted:Q', format=',.0f')]
        ).properties(height=350)
        st.altair_chart(chart_roles, use_container_width=True)
    else:
        st.warning("Dataset not found for analytics.")


# ==========================================
# TAB 3: CAREER GROWTH SIMULATOR
# ==========================================
with tab3:
    st.markdown("### 💡 What-If Career & Salary Growth Simulator")
    st.write("Simulate how leveling up your seniority, transitioning to a larger company scale, or changing remote work policy impacts your market value.")
    
    if model is not None and encoder is not None and config is not None:
        sim_c1, sim_c2 = st.columns(2)
        
        with sim_c1:
            st.markdown("#### 📌 Current Baseline Profile")
            base_role = st.selectbox("Current Role", options=available_designations, index=available_designations.index("Data Scientist") if "Data Scientist" in available_designations else 0, key="sim_base_role")
            base_exp = st.selectbox("Current Seniority", ["EN", "MI", "SE"], index=1, format_func=lambda x: {"EN": "🌱 Junior / Entry", "MI": "⚡ Mid-Level", "SE": "🚀 Senior"}[x], key="sim_base_exp")
            base_size = st.selectbox("Current Company Scale", ["S", "M", "L"], index=0, format_func=lambda x: {"S": "Small (<50)", "M": "Medium (50-250)", "L": "Large (>250)"}[x], key="sim_base_size")
            base_loc = st.selectbox("Current Company HQ", options=available_company_locs, index=available_company_locs.index("US") if "US" in available_company_locs else 0, format_func=format_country, key="sim_base_loc")

        with sim_c2:
            st.markdown("#### 🎯 Target Upgraded Profile")
            target_exp = st.selectbox("Target Seniority", ["MI", "SE", "EX"], index=min(2, ["EN", "MI", "SE"].index(base_exp) + 1), format_func=lambda x: {"MI": "⚡ Mid-Level", "SE": "🚀 Senior", "EX": "👑 Executive / Lead"}[x], key="sim_tar_exp")
            target_size = st.selectbox("Target Company Scale", ["S", "M", "L"], index=2, format_func=lambda x: {"S": "Small (<50)", "M": "Medium (50-250)", "L": "Large (>250)"}[x], key="sim_tar_size")
            target_remote = st.selectbox("Target Remote Policy", [0, 50, 100], index=2, format_func=lambda x: {0: "🏢 Onsite", 50: "🔄 Hybrid", 100: "🌐 Full Remote"}[x], key="sim_tar_rem")
            
        # Calculation
        def get_pred_salary(r, e, s, rem, cloc, eloc):
            df_s = pd.DataFrame([{'Designation': r, 'Experience': e, 'Employment_Status': 'FT', 'Company_Size': s, 'Remote_Working_Ratio': rem, 'Company_Location': cloc, 'Employee_Location': eloc, 'Working_Year': 2022}])
            fe = advanced_feature_engineering(df_s)
            rdy = fe[config.get('feature_cols', fe.columns.tolist())]
            enc = encoder.transform(rdy)
            p_log = model.predict(enc)[0]
            return float(np.expm1(p_log))
            
        base_sal = get_pred_salary(base_role, base_exp, base_size, 50, base_loc, base_loc)
        target_sal = get_pred_salary(base_role, target_exp, target_size, target_remote, base_loc, base_loc)
        diff_sal = target_sal - base_sal
        diff_pct = (diff_sal / base_sal) * 100
        
        st.divider()
        st.markdown(f"#### 🚀 Simulated Growth Output ({period_label})")
        s_res1, s_res2, s_res3 = st.columns(3)
        
        s_res1.metric(f"Current Baseline ({period_label})", format_currency_value(base_sal, selected_currency, selected_period))
        s_res2.metric(f"Target Upgraded ({period_label})", format_currency_value(target_sal, selected_currency, selected_period), delta=f"+{diff_pct:.1f}%")
        s_res3.metric(f"Estimated Increase ({period_label})", format_currency_value(diff_sal, selected_currency, selected_period), delta=f"+{diff_pct:.1f}%")