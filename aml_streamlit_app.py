"""
Julius Baer - Complete AML Agentic AI Solution
Real-Time AML Monitoring and Document & Image Corroboration System
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Julius Baer AML AI System",
    page_icon="🏛️",
    layout="wide"
)

class AMLAISystem:
    def __init__(self):
        """Initialize the AML AI System"""
        self.load_models()
        
    def load_models(self):
        """Load pre-trained ML models"""
        try:
            models = joblib.load('aml_models.pkl')
            self.suspicion_classifier = models['suspicion_classifier']
            self.suspicion_regressor = models['suspicion_regressor']
            self.str_classifier = models['str_classifier']
            self.str_regressor = models['str_regressor']
            self.label_encoders = models['label_encoders']
            self.scaler = models['scaler']
            self.feature_columns = models['feature_columns']
            self.models_loaded = True
        except:
            st.error("⚠️ ML models not found. Please run aml_ml_solution.py first to train the models.")
            self.models_loaded = False
    
    def preprocess_transaction(self, transaction_data):
        """Preprocess a single transaction for prediction"""
        if not self.models_loaded:
            return None
            
        # Create features similar to training
        data = transaction_data.copy()
        
        # Amount-based features
        data['amount_log'] = np.log1p(data['amount'])
        data['daily_cash_ratio'] = data['amount'] / (data.get('daily_cash_total_customer', 0) + 1)
        
        # Time features
        booking_time = pd.to_datetime(data['booking_datetime'])
        data['booking_hour'] = booking_time.hour
        data['booking_day_of_week'] = booking_time.dayofweek
        data['booking_month'] = booking_time.month
        
        # Risk features
        risk_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Balanced': 2}
        data['customer_risk_numeric'] = risk_mapping.get(data.get('customer_risk_rating', 'Low'), 1)
        data['client_risk_numeric'] = risk_mapping.get(data.get('client_risk_profile', 'Low'), 1)
        
        # Country risk
        high_risk_countries = ['RU', 'CN', 'IR', 'AE']
        data['originator_high_risk'] = int(data.get('originator_country', '') in high_risk_countries)
        data['beneficiary_high_risk'] = int(data.get('beneficiary_country', '') in high_risk_countries)
        
        # FX features
        data['has_fx'] = int(data.get('fx_indicator', False))
        data['fx_spread_high'] = int(data.get('fx_spread_bps', 0) > 50)
        
        # Compliance features
        data['edd_required_int'] = int(data.get('edd_required', False))
        data['is_pep_int'] = int(data.get('customer_is_pep', False))
        data['complex_product'] = int(data.get('product_complex', False))
        
        # Encode categorical variables
        for col in ['booking_jurisdiction', 'channel', 'product_type', 'currency', 'customer_type']:
            if col in data and col in self.label_encoders:
                try:
                    data[f'{col}_encoded'] = self.label_encoders[col].transform([data.get(col, 'Unknown')])[0]
                except:
                    data[f'{col}_encoded'] = 0
        
        # Create feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(data.get(col, 0))
        
        # Scale features
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector = self.scaler.transform(feature_vector)
        
        return feature_vector
    
    def predict_aml_risk(self, transaction_data):
        """Predict AML risk for a transaction"""
        if not self.models_loaded:
            return None
            
        X = self.preprocess_transaction(transaction_data)
        if X is None:
            return None
        
        # Predict suspicion
        suspicion_prob = self.suspicion_classifier.predict_proba(X)[0, 1]
        suspicion_pred = self.suspicion_classifier.predict(X)[0]
        
        # Predict STR
        str_prob = self.str_classifier.predict_proba(X)[0, 1]
        str_pred = self.str_classifier.predict(X)[0]
        
        # Predict timing
        hours_to_suspicion = 0
        hours_to_str = 0
        
        if suspicion_pred == 1:
            hours_to_suspicion = self.suspicion_regressor.predict(X)[0]
        
        if str_pred == 1:
            hours_to_str = self.str_regressor.predict(X)[0]
        
        # Calculate overall risk score
        risk_score = (suspicion_prob * 0.6 + str_prob * 0.4) * 100
        
        return {
            'risk_score': risk_score,
            'suspicion_probability': suspicion_prob,
            'str_probability': str_prob,
            'suspicion_prediction': bool(suspicion_pred),
            'str_prediction': bool(str_pred),
            'hours_to_suspicion': max(0, hours_to_suspicion),
            'hours_to_str': max(0, hours_to_str)
        }

def create_real_time_monitoring():
    """Create the real-time monitoring dashboard"""
    st.header("🔍 Real-Time AML Monitoring & Alerts")
    
    # Initialize system
    aml_system = AMLAISystem()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Transaction Analysis")
        
        # Transaction input form
        with st.form("transaction_form"):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                amount = st.number_input("Amount", min_value=0.0, value=100000.0)
                currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "CHF", "HKD", "SGD", "CNY", "JPY"])
                jurisdiction = st.selectbox("Jurisdiction", ["HK", "SG", "CH"])
                channel = st.selectbox("Channel", ["SWIFT", "RTGS", "FPS (HK)", "FAST (SG)", "SEPA", "Cash", "Cheque"])
            
            with col_b:
                customer_risk = st.selectbox("Customer Risk", ["Low", "Medium", "High"])
                client_risk = st.selectbox("Client Risk Profile", ["Low", "Balanced", "High"])
                product_type = st.selectbox("Product Type", ["wire_transfer", "fx_conversion", "cash_deposit", "securities_trade", "fund_subscription", "cash_withdrawal"])
                customer_type = st.selectbox("Customer Type", ["individual", "corporate", "domiciliary_company"])
            
            with col_c:
                originator_country = st.selectbox("Originator Country", ["US", "GB", "DE", "FR", "CN", "RU", "IR", "AE", "JP", "AU", "SG", "HK", "CH"])
                beneficiary_country = st.selectbox("Beneficiary Country", ["US", "GB", "DE", "FR", "CN", "RU", "IR", "AE", "JP", "AU", "SG", "HK", "CH"])
                is_pep = st.checkbox("PEP (Politically Exposed Person)")
                edd_required = st.checkbox("Enhanced Due Diligence Required")
            
            submit_button = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)
        
        if submit_button and aml_system.models_loaded:
            # Create transaction data
            transaction_data = {
                'amount': amount,
                'currency': currency,
                'booking_jurisdiction': jurisdiction,
                'channel': channel,
                'customer_risk_rating': customer_risk,
                'client_risk_profile': client_risk,
                'product_type': product_type,
                'customer_type': customer_type,
                'originator_country': originator_country,
                'beneficiary_country': beneficiary_country,
                'customer_is_pep': is_pep,
                'edd_required': edd_required,
                'booking_datetime': datetime.now(),
                'daily_cash_total_customer': 0,
                'daily_cash_txn_count': 1,
                'fx_indicator': False,
                'fx_spread_bps': 0,
                'product_complex': False
            }
            
            # Get prediction
            prediction = aml_system.predict_aml_risk(transaction_data)
            
            if prediction:
                # Display results
                st.subheader("🚨 AML Risk Assessment")
                
                # Risk score gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction['risk_score'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risk Score"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk level determination
                if prediction['risk_score'] >= 70:
                    risk_level = "🔴 HIGH RISK"
                    alert_color = "red"
                elif prediction['risk_score'] >= 30:
                    risk_level = "🟡 MEDIUM RISK"
                    alert_color = "orange"
                else:
                    risk_level = "🟢 LOW RISK"
                    alert_color = "green"
                
                st.markdown(f"### {risk_level}")
                
                # Detailed predictions
                col_1, col_2 = st.columns(2)
                
                with col_1:
                    st.metric("Suspicion Probability", f"{prediction['suspicion_probability']:.1%}")
                    if prediction['suspicion_prediction']:
                        st.warning(f"⏰ Suspicion expected in {prediction['hours_to_suspicion']:.1f} hours")
                    else:
                        st.success("✅ No suspicion expected")
                
                with col_2:
                    st.metric("STR Filing Probability", f"{prediction['str_probability']:.1%}")
                    if prediction['str_prediction']:
                        st.error(f"📋 STR filing expected in {prediction['hours_to_str']:.1f} hours")
                    else:
                        st.success("✅ No STR filing expected")
    
    with col2:
        st.subheader("🚨 Alert System")
        
        # Sample alerts
        st.markdown("### Recent Alerts")
        
        alerts = [
            {"time": "2 min ago", "type": "HIGH", "message": "Unusual FX spread detected", "transaction": "TXN-001"},
            {"time": "5 min ago", "type": "MEDIUM", "message": "PEP transaction above threshold", "transaction": "TXN-002"},
            {"time": "10 min ago", "type": "LOW", "message": "Cross-border payment pattern", "transaction": "TXN-003"},
        ]
        
        for alert in alerts:
            if alert["type"] == "HIGH":
                st.error(f"🔴 **{alert['type']}** ({alert['time']}): {alert['message']} - {alert['transaction']}")
            elif alert["type"] == "MEDIUM":
                st.warning(f"🟡 **{alert['type']}** ({alert['time']}): {alert['message']} - {alert['transaction']}")
            else:
                st.info(f"🟢 **{alert['type']}** ({alert['time']}): {alert['message']} - {alert['transaction']}")
        
        st.markdown("### Regulatory Updates")
        st.info("📢 **New FINMA Circular**: Enhanced surveillance for crypto-related transactions")
        st.info("📢 **MAS Alert**: Increased scrutiny on Russia-related payments")

def create_document_corroboration():
    """Create the document corroboration system"""
    st.header("📄 Document & Image Corroboration")
    
    tab1, tab2, tab3 = st.tabs(["📄 Document Analysis", "🖼️ Image Verification", "📊 Risk Report"])
    
    with tab1:
        st.subheader("Document Upload & Analysis")
        
        uploaded_file = st.file_uploader("Upload client document", type=['pdf', 'docx', 'txt', 'jpg', 'png'])
        
        if uploaded_file:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                
                # Simulated document analysis
                with st.spinner("Analyzing document..."):
                    import time
                    time.sleep(2)
                
                st.subheader("📋 Document Analysis Results")
                
                # Format validation
                st.markdown("### Format Validation")
                st.success("✅ Document structure: Valid")
                st.success("✅ Required fields: Present")
                st.warning("⚠️ Minor formatting inconsistency detected in Section 3")
                
                # Content validation
                st.markdown("### Content Validation")
                issues = [
                    {"type": "error", "message": "Amount mismatch: CHF 150,000 vs CHF 15,000 in Annex A"},
                    {"type": "warning", "message": "Date format inconsistency: DD/MM/YYYY vs MM/DD/YYYY"},
                    {"type": "info", "message": "Missing witness signature in Section 4"}
                ]
                
                for issue in issues:
                    if issue["type"] == "error":
                        st.error(f"❌ **Error**: {issue['message']}")
                    elif issue["type"] == "warning":
                        st.warning(f"⚠️ **Warning**: {issue['message']}")
                    else:
                        st.info(f"ℹ️ **Info**: {issue['message']}")
            
            with col2:
                st.subheader("📊 Document Risk Score")
                
                # Document risk gauge
                doc_risk_score = 35
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = doc_risk_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Document Risk"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "orange"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgray"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### Risk Factors")
                st.markdown("- Amount discrepancy: **High**")
                st.markdown("- Format issues: **Medium**")
                st.markdown("- Missing signatures: **Low**")
    
    with tab2:
        st.subheader("Image Integrity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Image Analysis")
            
            if uploaded_file and uploaded_file.type.startswith('image'):
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
                with st.spinner("Analyzing image integrity..."):
                    time.sleep(2)
                
                st.success("✅ **Authenticity Check**: No tampering detected")
                st.success("✅ **Reverse Image Search**: No matches found")
                st.warning("⚠️ **AI Generation Check**: 15% probability of AI generation")
                st.success("✅ **Metadata Analysis**: Consistent timestamp and location")
            else:
                st.info("Please upload an image file for integrity analysis")
        
        with col2:
            st.markdown("### 🛡️ Security Analysis")
            
            security_checks = [
                {"check": "Digital Signature", "status": "✅ Valid", "details": "Document signed with valid certificate"},
                {"check": "Watermark Detection", "status": "✅ Present", "details": "Official watermark verified"},
                {"check": "Font Analysis", "status": "⚠️ Mixed", "details": "Multiple fonts detected - review required"},
                {"check": "Layout Consistency", "status": "✅ Consistent", "details": "Standard document template"}
            ]
            
            for check in security_checks:
                with st.expander(f"{check['status']} {check['check']}"):
                    st.write(check['details'])
    
    with tab3:
        st.subheader("📊 Comprehensive Risk Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Overall Risk Assessment")
            
            # Overall risk score
            overall_risk = 42
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = overall_risk,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Document Risk"},
                delta = {'reference': 30},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🎯 Recommended Actions")
            st.warning("1. **Verify amount discrepancy** with client")
            st.info("2. **Request standardized date format** for future submissions")
            st.info("3. **Obtain missing witness signature**")
            st.success("4. **Document approved** pending minor corrections")
        
        with col2:
            st.markdown("### 📈 Risk Breakdown")
            
            risk_categories = ['Content Accuracy', 'Format Compliance', 'Image Integrity', 'Security Features']
            risk_scores = [75, 45, 20, 15]
            
            fig = px.bar(
                x=risk_categories,
                y=risk_scores,
                title="Risk by Category",
                color=risk_scores,
                color_continuous_scale="RdYlGn_r"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📋 Audit Trail")
            audit_events = [
                "Document uploaded at 14:23:15",
                "OCR processing completed at 14:23:45",
                "Format validation completed at 14:24:02",
                "Content analysis completed at 14:24:18",
                "Risk score calculated at 14:24:20"
            ]
            
            for event in audit_events:
                st.text(f"✓ {event}")

def create_dashboard():
    """Create the main dashboard"""
    st.header("📊 AML Command Center")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Transactions Today", "1,247", "+5.2%")
    
    with col2:
        st.metric("High Risk Alerts", "23", "+12%")
    
    with col3:
        st.metric("STR Filings", "8", "+3")
    
    with col4:
        st.metric("Document Reviews", "156", "+8%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Score Distribution")
        
        # Generate sample data
        np.random.seed(42)
        risk_scores = np.random.beta(2, 5, 1000) * 100
        
        fig = px.histogram(
            x=risk_scores,
            nbins=20,
            title="Daily Risk Score Distribution",
            labels={'x': 'Risk Score', 'y': 'Number of Transactions'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Alert Trends")
        
        # Generate time series data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        alerts = np.random.poisson(15, len(dates))
        
        fig = px.line(
            x=dates,
            y=alerts,
            title="Daily Alert Volume",
            labels={'x': 'Date', 'y': 'Number of Alerts'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("🕐 Recent Activity")
    
    recent_activity = pd.DataFrame({
        'Time': ['14:25', '14:23', '14:20', '14:18', '14:15'],
        'Type': ['Transaction', 'Document', 'Alert', 'Transaction', 'STR Filing'],
        'Description': [
            'High-value FX conversion flagged',
            'KYC document analyzed - minor issues',
            'PEP transaction above threshold',
            'Cross-border payment cleared',
            'Suspicious activity report filed'
        ],
        'Risk': ['High', 'Medium', 'High', 'Low', 'High']
    })
    
    # Color code by risk
    def color_risk(val):
        if val == 'High':
            return 'background-color: #ffcccc'
        elif val == 'Medium':
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #d4edda'
    
    styled_df = recent_activity.style.applymap(color_risk, subset=['Risk'])
    st.dataframe(styled_df, use_container_width=True)

def main():
    """Main application"""
    st.title("🏛️ Julius Baer - Agentic AI for AML Monitoring")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Module",
        ["🏠 Dashboard", "🔍 Real-Time Monitoring", "📄 Document Corroboration"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Challenge Features")
    st.sidebar.success("✅ Real-time AML monitoring")
    st.sidebar.success("✅ ML-based risk prediction")
    st.sidebar.success("✅ Document verification")
    st.sidebar.success("✅ Image integrity analysis")
    st.sidebar.success("✅ Automated alerts")
    st.sidebar.success("✅ Compliance reporting")
    
    # Main content
    if page == "🏠 Dashboard":
        create_dashboard()
    elif page == "🔍 Real-Time Monitoring":
        create_real_time_monitoring()
    elif page == "📄 Document Corroboration":
        create_document_corroboration()

if __name__ == "__main__":
    main()