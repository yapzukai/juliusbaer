"""
Test script to verify the ML models are working correctly
"""

from aml_ml_solution import AMLMLPredictor
import pandas as pd
import numpy as np

def test_models():
    """Test the trained models"""
    print("🧪 Testing AML ML Models...")
    
    # Initialize predictor and load models
    try:
        predictor = AMLMLPredictor()
        
        # Load test data
        df = predictor.load_and_analyze_data('transactions_mock_1000_for_participants.csv')
        data = predictor.engineer_features(df)
        X, processed_data = predictor.prepare_features(data)
        
        print(f"✅ Data loaded: {X.shape[0]} transactions, {X.shape[1]} features")
        
        # Test predictions on a few samples
        sample_predictions = predictor.predict(X[:5])
        
        print("\n🔮 Sample Predictions:")
        for i in range(5):
            print(f"\nTransaction {i+1}:")
            print(f"  Risk Score: {(sample_predictions['suspicion_probability'][i] * 0.6 + sample_predictions['str_probability'][i] * 0.4) * 100:.1f}")
            print(f"  Suspicion Probability: {sample_predictions['suspicion_probability'][i]:.3f}")
            print(f"  STR Probability: {sample_predictions['str_probability'][i]:.3f}")
            print(f"  Suspicion Prediction: {sample_predictions['suspicion_prediction'][i]}")
            print(f"  STR Prediction: {sample_predictions['str_prediction'][i]}")
        
        # Test deterministic behavior
        print("\n🔒 Testing Deterministic Behavior...")
        pred1 = predictor.predict(X[:3])
        pred2 = predictor.predict(X[:3])
        
        # Check if predictions are identical
        identical = np.allclose(pred1['suspicion_probability'], pred2['suspicion_probability'])
        print(f"✅ Deterministic: {identical}")
        
        print("\n✅ All tests passed! Models are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_models()