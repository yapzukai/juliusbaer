"""
Julius Baer AML ML Solution
Deterministic Machine Learning models to predict AML suspicion and STR filing times
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib

# Set random seeds for deterministic results
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class AMLMLPredictor:
    def __init__(self):
        self.suspicion_model = None
        self.str_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_and_analyze_data(self, file_path):
        """Load and analyze the dataset"""
        print("Loading AML transaction data...")
        df = pd.read_csv(file_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Analyze target variables
        print("\n=== TARGET ANALYSIS ===")
        print("Last two columns (targets):")
        print(f"Column -2: {df.columns[-2]} - {df.iloc[:, -2].name}")
        print(f"Column -1: {df.columns[-1]} - {df.iloc[:, -1].name}")
        
        suspicion_col = df.columns[-2]  # suspicion_determined_datetime
        str_col = df.columns[-1]        # str_filed_datetime
        
        print(f"\nSuspicion column ({suspicion_col}):")
        print(f"  Non-null values: {df[suspicion_col].notna().sum()}/{len(df)}")
        print(f"  Sample values: {df[suspicion_col].dropna().head().tolist()}")
        
        print(f"\nSTR filed column ({str_col}):")
        print(f"  Non-null values: {df[str_col].notna().sum()}/{len(df)}")
        print(f"  Sample values: {df[str_col].dropna().head().tolist()}")
        
        return df
    
    def engineer_features(self, df):
        """Create features for ML models"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Convert datetime columns
        datetime_cols = ['booking_datetime', 'value_date', 'kyc_last_completed', 'kyc_due_date']
        for col in datetime_cols:
            if col in data.columns:
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                except:
                    pass
        
        # Feature engineering
        print("Creating new features...")
        
        # Amount-based features
        data['amount_log'] = np.log1p(data['amount'])
        data['daily_cash_ratio'] = data['amount'] / (data['daily_cash_total_customer'] + 1)
        
        # Time-based features
        if 'booking_datetime' in data.columns:
            data['booking_hour'] = pd.to_datetime(data['booking_datetime']).dt.hour
            data['booking_day_of_week'] = pd.to_datetime(data['booking_datetime']).dt.dayofweek
            data['booking_month'] = pd.to_datetime(data['booking_datetime']).dt.month
        
        # Risk-related features
        risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Balanced': 2}
        data['customer_risk_numeric'] = data['customer_risk_rating'].map(risk_mapping)
        data['client_risk_numeric'] = data['client_risk_profile'].map(risk_mapping)
        
        # Currency and country diversity
        high_risk_countries = ['RU', 'CN', 'IR', 'AE']
        data['originator_high_risk'] = data['originator_country'].isin(high_risk_countries).astype(int)
        data['beneficiary_high_risk'] = data['beneficiary_country'].isin(high_risk_countries).astype(int)
        
        # FX features
        data['has_fx'] = data['fx_indicator'].astype(int)
        data['fx_spread_high'] = (data['fx_spread_bps'] > 50).astype(int)
        
        # Compliance features
        data['edd_required_int'] = data['edd_required'].astype(int)
        data['is_pep_int'] = data['customer_is_pep'].astype(int)
        data['complex_product'] = data['product_complex'].astype(int)
        
        # Create target variables
        # For suspicion: 1 if suspicion_determined_datetime exists, 0 otherwise
        data['has_suspicion'] = data['suspicion_determined_datetime'].notna().astype(int)
        
        # For STR: 1 if str_filed_datetime exists, 0 otherwise  
        data['has_str'] = data['str_filed_datetime'].notna().astype(int)
        
        # Time to suspicion (hours from booking to suspicion)
        suspicion_times = pd.to_datetime(data['suspicion_determined_datetime'], errors='coerce')
        booking_times = pd.to_datetime(data['booking_datetime'], errors='coerce')
        data['hours_to_suspicion'] = (suspicion_times - booking_times).dt.total_seconds() / 3600
        
        # Time to STR filing (hours from booking to STR)
        str_times = pd.to_datetime(data['str_filed_datetime'], errors='coerce')
        data['hours_to_str'] = (str_times - booking_times).dt.total_seconds() / 3600
        
        print(f"Suspicion cases: {data['has_suspicion'].sum()}/{len(data)}")
        print(f"STR filed cases: {data['has_str'].sum()}/{len(data)}")
        
        return data
    
    def prepare_features(self, data):
        """Prepare features for ML models"""
        print("\n=== FEATURE PREPARATION ===")
        
        # Select features for modeling
        feature_cols = [
            'amount', 'amount_log', 'daily_cash_ratio', 'daily_cash_total_customer', 'daily_cash_txn_count',
            'booking_hour', 'booking_day_of_week', 'booking_month',
            'customer_risk_numeric', 'client_risk_numeric',
            'originator_high_risk', 'beneficiary_high_risk',
            'has_fx', 'fx_spread_bps', 'fx_spread_high',
            'edd_required_int', 'is_pep_int', 'complex_product'
        ]
        
        # Add categorical features with encoding
        categorical_cols = ['booking_jurisdiction', 'channel', 'product_type', 'currency', 'customer_type']
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data[col].fillna('Unknown'))
                else:
                    data[f'{col}_encoded'] = self.label_encoders[col].transform(data[col].fillna('Unknown'))
                feature_cols.append(f'{col}_encoded')
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in data.columns]
        
        print(f"Selected {len(feature_cols)} features for modeling")
        self.feature_columns = feature_cols
        
        # Prepare feature matrix
        X = data[feature_cols].copy()
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        return X_scaled, data
    
    def train_suspicion_model(self, X, y_suspicion, y_hours_to_suspicion):
        """Train model to predict suspicion determination"""
        print("\n=== TRAINING SUSPICION MODEL ===")
        
        # Classification model: Will there be suspicion?
        self.suspicion_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        # Regression model: When will suspicion be determined? (for cases where suspicion occurs)
        self.suspicion_regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        # Train classification model on all data
        X_train, X_test, y_train, y_test = train_test_split(X, y_suspicion, test_size=0.2, random_state=RANDOM_STATE, stratify=y_suspicion)
        
        self.suspicion_classifier.fit(X_train, y_train)
        suspicion_pred = self.suspicion_classifier.predict(X_test)
        
        print("Suspicion Classification Results:")
        print(classification_report(y_test, suspicion_pred))
        
        # Train regression model on cases with suspicion
        suspicion_mask = y_hours_to_suspicion.notna()
        if suspicion_mask.sum() > 0:
            X_suspicion = X[suspicion_mask]
            y_hours_suspicion = y_hours_to_suspicion[suspicion_mask]
            
            X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
                X_suspicion, y_hours_suspicion, test_size=0.2, random_state=RANDOM_STATE
            )
            
            self.suspicion_regressor.fit(X_reg_train, y_reg_train)
            hours_pred = self.suspicion_regressor.predict(X_reg_test)
            
            print(f"\nSuspicion Timing Regression Results:")
            print(f"  MAE: {mean_absolute_error(y_reg_test, hours_pred):.2f} hours")
            print(f"  RMSE: {np.sqrt(mean_squared_error(y_reg_test, hours_pred)):.2f} hours")
    
    def train_str_model(self, X, y_str, y_hours_to_str):
        """Train model to predict STR filing"""
        print("\n=== TRAINING STR MODEL ===")
        
        # Classification model: Will STR be filed?
        self.str_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        # Regression model: When will STR be filed?
        self.str_regressor = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        # Train classification model
        X_train, X_test, y_train, y_test = train_test_split(X, y_str, test_size=0.2, random_state=RANDOM_STATE, stratify=y_str)
        
        self.str_classifier.fit(X_train, y_train)
        str_pred = self.str_classifier.predict(X_test)
        
        print("STR Classification Results:")
        print(classification_report(y_test, str_pred))
        
        # Train regression model on cases with STR
        str_mask = y_hours_to_str.notna()
        if str_mask.sum() > 0:
            X_str = X[str_mask]
            y_hours_str = y_hours_to_str[str_mask]
            
            X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
                X_str, y_hours_str, test_size=0.2, random_state=RANDOM_STATE
            )
            
            self.str_regressor.fit(X_reg_train, y_reg_train)
            hours_pred = self.str_regressor.predict(X_reg_test)
            
            print(f"\nSTR Timing Regression Results:")
            print(f"  MAE: {mean_absolute_error(y_reg_test, hours_pred):.2f} hours")
            print(f"  RMSE: {np.sqrt(mean_squared_error(y_reg_test, hours_pred)):.2f} hours")
    
    def predict(self, X):
        """Make predictions for new data"""
        # Predict suspicion
        suspicion_prob = self.suspicion_classifier.predict_proba(X)[:, 1]
        suspicion_pred = self.suspicion_classifier.predict(X)
        
        # Predict hours to suspicion for cases with suspicion
        hours_to_suspicion = np.zeros(len(X))
        suspicion_mask = suspicion_pred == 1
        if suspicion_mask.sum() > 0:
            hours_to_suspicion[suspicion_mask] = self.suspicion_regressor.predict(X[suspicion_mask])
        
        # Predict STR
        str_prob = self.str_classifier.predict_proba(X)[:, 1] 
        str_pred = self.str_classifier.predict(X)
        
        # Predict hours to STR for cases with STR
        hours_to_str = np.zeros(len(X))
        str_mask = str_pred == 1
        if str_mask.sum() > 0:
            hours_to_str[str_mask] = self.str_regressor.predict(X[str_mask])
        
        return {
            'suspicion_probability': suspicion_prob,
            'suspicion_prediction': suspicion_pred,
            'hours_to_suspicion': hours_to_suspicion,
            'str_probability': str_prob, 
            'str_prediction': str_pred,
            'hours_to_str': hours_to_str
        }
    
    def get_feature_importance(self):
        """Get feature importance from trained models"""
        importance_data = {}
        
        if self.suspicion_classifier:
            importance_data['suspicion_classification'] = dict(zip(
                self.feature_columns, 
                self.suspicion_classifier.feature_importances_
            ))
        
        if self.str_classifier:
            importance_data['str_classification'] = dict(zip(
                self.feature_columns,
                self.str_classifier.feature_importances_
            ))
            
        return importance_data
    
    def save_models(self, path_prefix='aml_models'):
        """Save trained models"""
        models = {
            'suspicion_classifier': self.suspicion_classifier,
            'suspicion_regressor': self.suspicion_regressor,
            'str_classifier': self.str_classifier,
            'str_regressor': self.str_regressor,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(models, f'{path_prefix}.pkl')
        print(f"Models saved to {path_prefix}.pkl")


def main():
    """Main execution function"""
    print("=== JULIUS BAER AML ML SOLUTION ===")
    
    # Initialize predictor
    predictor = AMLMLPredictor()
    
    # Load and analyze data
    df = predictor.load_and_analyze_data('transactions_mock_1000_for_participants.csv')
    
    # Engineer features
    data = predictor.engineer_features(df)
    
    # Prepare features
    X, processed_data = predictor.prepare_features(data)
    
    # Prepare target variables
    y_suspicion = processed_data['has_suspicion']
    y_hours_to_suspicion = processed_data['hours_to_suspicion']
    y_str = processed_data['has_str']
    y_hours_to_str = processed_data['hours_to_str']
    
    # Train models
    predictor.train_suspicion_model(X, y_suspicion, y_hours_to_suspicion)
    predictor.train_str_model(X, y_str, y_hours_to_str)
    
    # Show feature importance
    print("\n=== FEATURE IMPORTANCE ===")
    importance = predictor.get_feature_importance()
    
    for model_name, features in importance.items():
        print(f"\n{model_name.upper()}:")
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:10]
        for feature, importance_val in sorted_features:
            print(f"  {feature}: {importance_val:.4f}")
    
    # Make sample predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    sample_predictions = predictor.predict(X[:10])
    
    for i in range(5):
        print(f"\nTransaction {i+1}:")
        print(f"  Suspicion Probability: {sample_predictions['suspicion_probability'][i]:.3f}")
        print(f"  Suspicion Prediction: {sample_predictions['suspicion_prediction'][i]}")
        print(f"  Hours to Suspicion: {sample_predictions['hours_to_suspicion'][i]:.1f}")
        print(f"  STR Probability: {sample_predictions['str_probability'][i]:.3f}")
        print(f"  STR Prediction: {sample_predictions['str_prediction'][i]}")
        print(f"  Hours to STR: {sample_predictions['hours_to_str'][i]:.1f}")
    
    # Save models
    predictor.save_models()
    
    print("\n=== MODEL TRAINING COMPLETE ===")
    print("Models are deterministic and will produce the same results on each run.")
    return predictor


if __name__ == "__main__":
    predictor = main()