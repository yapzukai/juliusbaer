#!/usr/bin/env python3
"""
Julius Baer AML Challenge - Main Application Runner
This script sets up and runs the complete AML AI solution.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required files and models exist"""
    required_files = [
        'transactions_mock_1000_for_participants.csv',
        'aml_ml_solution.py',
        'aml_streamlit_app.py'
    ]
    
    print("🔍 Checking requirements...")
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing required file: {file}")
            return False
        else:
            print(f"✅ Found: {file}")
    
    return True

def train_models():
    """Train the ML models"""
    print("\n🤖 Training ML models...")
    
    if not os.path.exists('aml_models.pkl'):
        print("Training new models...")
        result = subprocess.run([sys.executable, 'aml_ml_solution.py'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Models trained successfully!")
        else:
            print("❌ Model training failed:")
            print(result.stderr)
            return False
    else:
        print("✅ Pre-trained models found!")
    
    return True

def run_streamlit_app():
    """Run the Streamlit application"""
    print("\n🚀 Starting AML AI System...")
    print("Opening web application at http://localhost:8501")
    
    # Run Streamlit
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'aml_streamlit_app.py'])

def main():
    """Main execution function"""
    print("=" * 60)
    print("🏛️  JULIUS BAER - AML AGENTIC AI SOLUTION")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please ensure all required files are present.")
        sys.exit(1)
    
    # Train models
    if not train_models():
        print("\n❌ Model training failed. Please check the error messages above.")
        sys.exit(1)
    
    # Run the application
    run_streamlit_app()

if __name__ == "__main__":
    main()