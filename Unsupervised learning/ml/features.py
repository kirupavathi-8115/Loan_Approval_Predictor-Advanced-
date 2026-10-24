import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X_out = X.copy()
        
        # Drop customer_id if present
        if 'customer_id' in X_out.columns:
            X_out = X_out.drop(columns=['customer_id'])
            
        # Log1p transformations to handle skewness
        for col in ['income', 'total_spend']:
            if col in X_out.columns:
                X_out[col] = np.log1p(X_out[col].clip(lower=0))
                
        # Fill missing values if any
        if X_out.isnull().values.any():
            X_out = X_out.fillna(X_out.median())
                
        return X_out
