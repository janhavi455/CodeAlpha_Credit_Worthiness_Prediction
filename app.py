import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Creditworthiness Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('creditworthiness_model.pkl')
        return model
    except:
        st.error("Model file not found. Please ensure 'creditworthiness_model.pkl' is in the same directory.")
        return None

# Preprocess input data
def preprocess_input(input_data, feature_names):
    """Convert input data to match training format"""
    df = pd.DataFrame([input_data])
    
    # Ensure correct column order and types
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value
    
    return df[feature_names]

# Main app
def main():
    # Header
    st.title("🏦 Creditworthiness Assessment Tool")
    st.markdown("""
    This tool predicts the likelihood of a borrower experiencing serious delinquency (90+ days past due) 
    in the next 2 years. Enter the applicant's financial information below to get a risk assessment.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Get feature names (adjust based on your actual feature names)
    feature_names = [
        'RevolvingUtilizationOfUnsecuredLines', 'age', 
        'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ]
    
    # Create input form
    with st.form("credit_app_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            age = st.slider("Age", 18, 100, 45)
            num_dependents = st.number_input("Number of Dependents", 0, 20, 0)
            
            st.subheader("Income & Debt")
            monthly_income = st.number_input("Monthly Income ($)", 0, 50000, 5000, step=100)
            debt_ratio = st.slider("Debt Ratio", 0.0, 10.0, 0.5, 0.1)
            revolv_util = st.slider("Revolving Utilization %", 0.0, 1.0, 0.3, 0.01)
        
        with col2:
            st.subheader("Credit History")
            past_due_30_59 = st.number_input("30-59 Days Past Due (last 2 years)", 0, 10, 0)
            past_due_60_89 = st.number_input("60-89 Days Past Due (last 2 years)", 0, 10, 0)
            past_due_90 = st.number_input("90+ Days Past Due (last 2 years)", 0, 10, 0)
            
            st.subheader("Credit Accounts")
            open_credit_lines = st.number_input("Number of Open Credit Lines/Loans", 0, 50, 5)
            real_estate_loans = st.number_input("Number of Real Estate Loans/Lines", 0, 10, 1)
        
        submitted = st.form_submit_button("🔍 Assess Creditworthiness")
    
    # When form is submitted
    if submitted:
        # Create input data dictionary
        input_data = {
            'RevolvingUtilizationOfUnsecuredLines': revolv_util,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': past_due_30_59,
            'DebtRatio': debt_ratio,
            'MonthlyIncome': monthly_income,
            'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
            'NumberOfTimes90DaysLate': past_due_90,
            'NumberRealEstateLoansOrLines': real_estate_loans,
            'NumberOfTime60-89DaysPastDueNotWorse': past_due_60_89,
            'NumberOfDependents': num_dependents
        }
        
        # Preprocess and predict
        try:
            input_df = preprocess_input(input_data, feature_names)
            probability = model.predict_proba(input_df)[0, 1]
            prediction = model.predict(input_df)[0]
            
            # Display results
            st.success("Assessment Complete!")
            
            # Create result columns
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.metric("Default Probability", f"{probability:.1%}")
            
            with res_col2:
                risk_level = "High Risk" if probability > 0.3 else "Medium Risk" if probability > 0.1 else "Low Risk"
                st.metric("Risk Level", risk_level)
            
            with res_col3:
                recommendation = "Decline" if probability > 0.4 else "Further Review" if probability > 0.2 else "Approve"
                st.metric("Recommendation", recommendation)
            
            # Risk gauge
            st.subheader("Risk Assessment Gauge")
            st.progress(float(probability))
            
            # Detailed explanation
            with st.expander("📊 Detailed Analysis"):
                st.write(f"**Probability of Serious Delinquency:** {probability:.2%}")
                st.write(f"**Prediction:** {'High Risk' if prediction == 1 else 'Low Risk'}")
                
                if probability > 0.5:
                    st.error("⚠️ High probability of default. Strongly consider declining.")
                elif probability > 0.2:
                    st.warning("⚠️ Moderate risk. Recommend further review and possibly higher interest rate.")
                else:
                    st.success("✅ Low risk profile. Good candidate for approval.")
            
            # Feature importance explanation
            with st.expander("🔍 Key Factors Influencing This Decision"):
                st.write("""
                - **High revolving utilization** increases risk
                - **Multiple past due incidents** significantly impact score
                - **Lower income** relative to debt increases risk
                - **Age and dependents** are moderate factors
                """)
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About This Tool")
        st.info("""
        This credit risk model predicts the probability of a borrower 
        being 90+ days delinquent in the next 2 years.
        
        **Model Performance:**
        - ROC AUC: 0.8594
        - Recall: 75.2%
        - Specificity: 80.4%
        """)
        
        st.header("Risk Thresholds")
        st.write("""
        - **Low Risk:** < 20% probability
        - **Medium Risk:** 20% - 40% probability  
        - **High Risk:** > 40% probability
        """)
        
        st.header("Data Privacy")
        st.warning("""
        All data entered is processed locally and not stored. 
        This is a demonstration tool for credit risk assessment.
        """)

# Run the app
if __name__ == "__main__":
    main()