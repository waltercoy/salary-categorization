import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ISO 2-letter Country code to full country name mapping
COUNTRY_NAME_MAP = {
    'AE': 'United Arab Emirates', 'AR': 'Argentina', 'AT': 'Austria', 'AU': 'Australia',
    'BE': 'Belgium', 'BG': 'Bulgaria', 'BO': 'Bolivia', 'BR': 'Brazil', 'CA': 'Canada',
    'CH': 'Switzerland', 'CL': 'Chile', 'CN': 'China', 'CO': 'Colombia', 'CZ': 'Czech Republic',
    'DE': 'Germany', 'DK': 'Denmark', 'DZ': 'Algeria', 'EE': 'Estonia', 'ES': 'Spain',
    'FR': 'France', 'GB': 'United Kingdom', 'GR': 'Greece', 'HK': 'Hong Kong', 'HN': 'Honduras',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IL': 'Israel', 'IN': 'India',
    'IQ': 'Iraq', 'IR': 'Iran', 'IT': 'Italy', 'JE': 'Jersey', 'JP': 'Japan',
    'KE': 'Kenya', 'LU': 'Luxembourg', 'MD': 'Moldova', 'MT': 'Malta', 'MX': 'Mexico',
    'MY': 'Malaysia', 'NG': 'Nigeria', 'NL': 'Netherlands', 'NZ': 'New Zealand',
    'PH': 'Philippines', 'PK': 'Pakistan', 'PL': 'Poland', 'PR': 'Puerto Rico',
    'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia', 'RU': 'Russia', 'SG': 'Singapore',
    'SI': 'Slovenia', 'TN': 'Tunisia', 'TR': 'Turkey', 'UA': 'Ukraine', 'US': 'United States',
    'VN': 'Vietnam', 'AS': 'American Samoa'
}

# Currency conversion rates relative to INR (Indian Rupee)
CURRENCY_RATES = {
    'USD': {'symbol': '$', 'rate': 0.012, 'name': 'US Dollar (USD)'},
    'MYR': {'symbol': 'RM', 'rate': 0.054, 'name': 'Malaysian Ringgit (MYR)'},
    'IDR': {'symbol': 'Rp', 'rate': 188.0, 'name': 'Indonesian Rupiah (IDR)'},
    'EUR': {'symbol': '€', 'rate': 0.011, 'name': 'Euro (EUR)'},
    'INR': {'symbol': '₹', 'rate': 1.0, 'name': 'Indian Rupee (INR)'}
}

class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder with m-estimate smoothing to prevent overfitting
    on high-cardinality features or sparse categorical groups.
    """
    def __init__(self, cols, m=10.0):
        self.cols = cols
        self.m = m
        self.maps = {}
        self.global_mean = 0.0
        
    def fit(self, X, y):
        self.global_mean = float(y.mean())
        for col in self.cols:
            grouped = pd.DataFrame({'feat': X[col], 'target': y}).groupby('feat')
            stats = grouped['target'].agg(['count', 'mean'])
            # Smoothing Formula: (count * mean + m * global_mean) / (count + m)
            smoothed = (stats['count'] * stats['mean'] + self.m * self.global_mean) / (stats['count'] + self.m)
            self.maps[col] = smoothed.to_dict()
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.cols:
            if col in self.maps:
                mapping = self.maps[col]
                X_out[col] = X_out[col].map(mapping).fillna(self.global_mean)
        return X_out

# Backward compatibility alias
TargetEncoder = SmoothedTargetEncoder


def advanced_feature_engineering(df_in):
    """
    Prepares structured features for both training and real-time inference.
    """
    df_eng = df_in.copy()
    
    # 1. Company Size Score (Ordinal: S=1, M=2, L=3)
    size_map = {'S': 1, 'M': 2, 'L': 3}
    df_eng['size_score'] = df_eng['Company_Size'].map(size_map).fillna(2)
    
    # 2. Experience Level Score (Ordinal: EN=1, MI=2, SE=3, EX=4)
    exp_map = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
    df_eng['exp_score'] = df_eng['Experience'].map(exp_map).fillna(2)
    
    # 3. Employment Status Score (PT=1, FL=2, CT=3, FT=4)
    emp_map = {'PT': 1, 'FL': 2, 'CT': 3, 'FT': 4}
    if 'Employment_Status' in df_eng.columns:
        df_eng['emp_score'] = df_eng['Employment_Status'].map(emp_map).fillna(4)
    else:
        df_eng['emp_score'] = 4  # default Full-time
        
    # 4. Remote Work Ratio Score
    df_eng['remote_score'] = df_eng['Remote_Working_Ratio']
    
    # 5. Same Country Interaction Flag
    df_eng['is_same_country'] = (df_eng['Company_Location'] == df_eng['Employee_Location']).astype(int)
    
    # 6. Working Year
    if 'Working_Year' not in df_eng.columns:
        df_eng['Working_Year'] = 2022
        
    return df_eng
