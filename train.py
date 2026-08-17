import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error, r2_score, f1_score
from sklearn.ensemble import HistGradientBoostingRegressor
from model_utils import SmoothedTargetEncoder, advanced_feature_engineering

def train_and_evaluate():
    print("=" * 65)
    print("=== STARTING SALARY PREDICTION MODEL TRAINING & OPTIMIZATION ===")
    print("=" * 65)
    
    # 1. Load Data
    csv_file = "Data_Science_Fields_Salary_Categorization.csv"
    print(f"[*] Loading dataset: {csv_file}")
    df = pd.read_csv(csv_file)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Clean Salary
    df['Salary_In_Rupees'] = df['Salary_In_Rupees'].astype(str).str.replace(',', '').astype(float)
    
    # 2. Target Discretization (Terciles)
    labels = ['Low', 'Medium', 'High']
    df['salary_category'] = pd.qcut(df['Salary_In_Rupees'], q=3, labels=labels, duplicates='drop')
    
    # 3. Feature Engineering
    print("[*] Executing feature engineering pipeline...")
    df_clean = advanced_feature_engineering(df)
    
    cols_to_encode = ['Designation', 'Company_Location', 'Employee_Location']
    num_cols = ['size_score', 'exp_score', 'emp_score', 'remote_score', 'is_same_country', 'Working_Year']
    feature_cols = num_cols + cols_to_encode
    
    X = df_clean[feature_cols]
    y_num = df_clean['Salary_In_Rupees']
    y_log = np.log1p(df_clean['Salary_In_Rupees']) # Log Target Transformation
    y_cat = df_clean['salary_category']
    
    # Thresholds
    t1 = float(y_num.quantile(0.33))
    t2 = float(y_num.quantile(0.66))
    
    # 4. Train-Test Split (Stratified on Category)
    print("[*] Splitting dataset (70% Train, 30% Test)...")
    X_train, X_test, y_train_log, y_test_log, y_train_cat, y_test_cat, y_train_orig, y_test_orig = train_test_split(
        X, y_log, y_cat, y_num, test_size=0.3, random_state=42, stratify=y_cat
    )
    
    # 5. Smoothed Target Encoding
    print("[*] Applying Smoothed Target Encoding (m=10.0)...")
    encoder = SmoothedTargetEncoder(cols=cols_to_encode, m=10.0)
    X_train_enc = encoder.fit_transform(X_train, y_train_log)
    X_test_enc = encoder.transform(X_test)
    
    # 6. Model Training (HistGradientBoostingRegressor)
    print("[*] Training HistGradientBoostingRegressor on log-salary target...")
    model = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.03,
        max_depth=6,
        l2_regularization=0.5,
        random_state=42
    )
    model.fit(X_train_enc, y_train_log)
    
    # 7. Predict & Transform back
    pred_log = model.predict(X_test_enc)
    pred_num = np.expm1(pred_log) # Invert log1p
    
    # Convert to Category
    def convert_to_cat(vals):
        res = []
        for x in vals:
            if x <= t1:
                res.append('Low')
            elif x <= t2:
                res.append('Medium')
            else:
                res.append('High')
        return np.array(res)
        
    pred_cat = convert_to_cat(pred_num)
    
    # 8. Evaluation Metrics
    acc = accuracy_score(y_test_cat, pred_cat)
    f1_weighted = f1_score(y_test_cat, pred_cat, average='weighted')
    mae = mean_absolute_error(y_test_orig, pred_num)
    r2 = r2_score(y_test_orig, pred_num)
    
    print("\n" + "=" * 65)
    print("=== MODEL EVALUATION METRICS AFTER OPTIMIZATION ===")
    print("=" * 65)
    print(f" -> Category Accuracy (Test Set) : {acc * 100:.2f}%")
    print(f" -> Weighted F1-Score            : {f1_weighted * 100:.2f}%")
    print(f" -> Mean Absolute Error (MAE)    : INR {mae:,.0f}")
    print(f" -> R2 Score (Regression)        : {r2:.4f}")
    print("\n[Detailed Classification Report]:")
    print(classification_report(y_test_cat, pred_cat))
    print("[Confusion Matrix (Low, Medium, High)]:")
    print(confusion_matrix(y_test_cat, pred_cat, labels=['Low', 'Medium', 'High']))
    
    # 9. Save Artifacts
    print("\n[*] Saving model artifacts...")
    joblib.dump(model, 'best_model.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    joblib.dump({
        't1': t1,
        't2': t2,
        'num_cols': num_cols,
        'cols_to_encode': cols_to_encode,
        'feature_cols': feature_cols,
        'accuracy': acc,
        'f1_score': f1_weighted
    }, 'config.pkl')
    
    print("[+] Model saved: best_model.pkl")
    print("[+] Encoder saved: encoder.pkl")
    print("[+] Config saved: config.pkl")
    print("\n[OK] Training completed successfully!")

if __name__ == '__main__':
    train_and_evaluate()
