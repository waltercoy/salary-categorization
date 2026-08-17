# Data Science Salary Categorization & Intelligence

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Optimized-brightgreen)

## 📌 Project Overview
This project builds an end-to-end Machine Learning pipeline and interactive web dashboard to predict and categorize the compensation of Data Science professionals globally into three tiers: **Low**, **Medium**, and **High**.

By leveraging **Log-Target Regressor modeling** (`HistGradientBoostingRegressor`) with **Smoothed Target Encoding** and ordinal feature engineering, the model achieves high accuracy and delivers interpretable compensation insights.

---

## 📊 Dataset Information

* **Source:** [Kaggle - Data Science Fields Salary Categorization](https://www.kaggle.com/datasets/whenamancodes/data-science-fields-salary-categorization)
* **Volume:** 607 records with 10 features.
* **Key Attributes:**
  * `Designation`: Job title (Data Scientist, Data Engineer, ML Engineer, etc.)
  * `Experience`: Seniority level (`EN` Entry, `MI` Mid, `SE` Senior, `EX` Executive)
  * `Employment_Status`: Full-time, Part-time, Contract, Freelance
  * `Company_Size`: Small (`S`), Medium (`M`), Large (`L`)
  * `Remote_Working_Ratio`: 0 (Onsite), 50 (Hybrid), 100 (Remote)
  * `Company_Location` & `Employee_Location`: ISO country codes
  * `Working_Year`: Market reporting year (2020 – 2022)
  * `Salary_In_Rupees`: Target continuous annual compensation

---

## ⚙️ Machine Learning Pipeline & Improvements

1. **Log-Transformation Target:**
   * Solves right-skewed salary distributions by training on $\ln(1 + \text{Salary})$, stabilizing gradients and improving mid-tier classification.
2. **Smoothed Target Encoding ($m$-estimate):**
   * Encodes high-cardinality features (`Designation`, `Company_Location`, `Employee_Location`) using empirical Bayesian smoothing ($m=10.0$) to eliminate overfitting on rare categories.
3. **Engineered Interaction Features:**
   * `is_same_country`: Cross-border vs domestic employment indicator.
   * `emp_score` & `size_score`: Ordinal mappings for company size and employment type.
4. **Quantile Discretization:**
   * Maps continuous predicted salaries into 3 balanced tiers (*Low, Medium, High*) using empirical terciles ($t_1 = 33\%$, $t_2 = 66\%$).

---

## 📈 Benchmark & Performance Results

| Model Architecture | Accuracy | Weighted F1 | $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :--- |
| **KNN Classifier ($K=8$)** | ~62.0% | ~61.5% | — | Baseline |
| **Decision Tree Classifier** | ~63.0% | ~62.8% | — | Overfitted |
| **Random Forest Baseline** | ~66.0% | ~66.0% | — | Previous Model |
| **HistGradientBoosting (Log Target + Smoothed TE)** | **71.58%** | **71.66%** | **0.4042** | 🏆 **Optimized Model** |

---

## 🚀 How to Run

### 1. Installation
```bash
# Clone repository
git clone https://github.com/waltercoy/salary-categorization.git
cd salary-categorization

# Install dependencies
pip install -r requirements.txt
```

### 2. Retrain the Model
You can retrain via one-click CLI or the notebook:
```bash
# Option A: Command-line automated training
python train.py

# Option B: Jupyter Notebook
jupyter notebook "salary_prediction.ipynb"
```

### 3. Launch Interactive Web App
```bash
streamlit run app.py
```

### 🌟 Web App Features:
* **🔮 Tab 1 - Salary Predictor:** Multi-currency support (**USD $, MYR RM, IDR Rp, EUR €, INR ₹**), payment period toggle (Monthly / Yearly), interactive form, and salary position gauges.
* **📈 Tab 2 - Market Insights:** Exploratory charts of average compensation by seniority, company scale, and top-paying roles.
* **💡 Tab 3 - Career Growth Simulator:** Interactive "What-If" simulator comparing current vs target seniority levels.

---

## 👤 Author
**ARIA FIRMANSYAH**
* LinkedIn: [Aria Firmansyah](https://www.linkedin.com/in/aria-firmansyah-0b1a87286/)
* Dataset: [Kaggle](https://www.kaggle.com/datasets/whenamancodes/data-science-fields-salary-categorization)