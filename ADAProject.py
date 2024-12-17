import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import json
import os

current_dir = os.path.dirname(__file__)
#hello_json_path = os.path.join(current_dir, 'hello.json')
machine_json_path = os.path.join(current_dir, 'machine.json')
model_path = os.path.join(current_dir, 'LOGREG_model.pkl')
data_path = os.path.join(current_dir, 'bank-additional.csv')

# Lottie animation loading function
def load_lottie_file(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)
    
@st.cache(allow_output_mutation=True)
def load_model():
    with st.spinner("Loading model..."):
        return joblib.load(model_path)

@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv(data_path)

@st.cache(allow_output_mutation=True)
def load_animation_m():
    return load_lottie_file(machine_json_path)

# Load animations, model and data
lottie_animation_machine = load_animation_m()
model = load_model()
data = load_data()

feature_info = """
### Feature Information:
- **age**: Age of the client.
- **job**: Type of job.
- **marital**: Marital status.
- **education**: Education level.
- **default**: Has credit in default? (yes, no).
- **housing**: Has housing loan? (yes, no).
- **loan**: Has personal loan? (yes, no).
- **contact**: Contact communication type.
- **month**: Last contact month of year.
- **day_of_week**: Last contact day of the week.
- **duration**: Last contact duration, in seconds.
- **campaign**: number of contacts performed during this campaign and for this client (numeric, includes last contact).
- **pdays**: Number of days that passed by after the client was last contacted from a previous campaign.
- **previous**: Number of contacts performed before this campaign and for this client.
- **poutcome**: Outcome of the previous marketing campaign.
- **emp.var.rate**: Employment variation rate.
- **cons.price.idx**: Consumer price index.
- **cons.conf.idx**: Consumer confidence index.
- **euribor3m**: Euribor 3 month rate.
- **nr.employed**: Number of employees.
- **Feature Creation --> campaign_previous_interaction**: Interaction between campaign and previous contacts.
- **y**: has the client subscribed a term deposit?
"""

# Upload trained model 
model = joblib.load(model_path)
data= pd.read_csv(data_path)

# Page selection using sidebar
st.sidebar.title("Page Navigator")
page = st.sidebar.selectbox("Choose a page", ["Home", "Data Visualization", "Machine Learning Prediction Model","Feedback"])

if page == "Home":
   st.title("Welcome to the Bank Marketing Campaign Prediction App")
   col1, col2 = st.columns(2)
   with col1:
    st.write("""
            This application allows you to predict the success of a bank marketing campaign 
            based on various features. Use the sidebar to navigate to the prediction page 
            and input the required features to get a prediction.
        """)
    
    #user_name = st.text_input("Can you enter your name here so I can address you?")
    #if user_name:
    #    st.write(f"Hello, {user_name}! Welcome to the app.")
    #    st.write("You can go to the navigation bar to explore the project we have created. Let's continue!")
