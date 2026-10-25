import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Sales Predictor App", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model.sav")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model 'model.sav': {e}")
    st.stop()

# App title and description
st.title("📈 Sales Prediction Dashboard")
st.write("""
Input your advertising budgets below to estimate predicted product sales. 
This application uses a trained Multiple Linear Regression model.
""")

# Sidebar / Input Form
st.subheader("Configure Advertising Budget")
col1, col2, col3 = st.columns(3)

with col1:
    tv = st.number_input("TV Budget ($)", min_value=0.0, max_value=1000.0, value=150.0, step=1.0)
with col2:
    radio = st.number_input("Radio Budget ($)", min_value=0.0, max_value=1000.0, value=25.0, step=1.0)
with col3:
    newspaper = st.number_input("Newspaper Budget ($)", min_value=0.0, max_value=1000.0, value=15.0, step=1.0)

# Predict Button
if st.button("Predict Sales", use_container_width=True):
    # Prepare the input features matching the exact column names
    input_data = pd.DataFrame({
        'TV': [tv],
        'Radio': [radio],
        'Newspaper': [newspaper]
    })
    
    # Generate Prediction
    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.success(f"### Predicted Sales: **{prediction:.2f}** units")
    
    # Expandable detail on formula
    with st.expander("View Model Coefficients Details"):
        st.write("The prediction is calculated using the following linear relationship:")
        st.code("Sales = (0.0441 * TV) + (0.1993 * Radio) + (0.0069 * Newspaper) + 2.7089")

# Write the requirements.txt file
with open("requirements.txt", "w") as f:
    f.write("""streamlit\nscikit-learn\npandas\nnumpy\njoblib\n""")

print("Successfully created app.py and requirements.txt!")
