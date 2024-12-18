import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import json
import os

current_dir = os.path.dirname(__file__)
hello_json_path = os.path.join(current_dir, 'hello.json')
machine_json_path = os.path.join(current_dir, 'machine.json')
model_path = os.path.join(current_dir, 'LOGREG_model.pkl')
data_path = os.path.join(current_dir, 'bank-additional.csv')

# Lottie animation loading function
def load_lottie_file(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)
    
@st.cache_resource()
def load_model():
    with st.spinner("Loading model..."):
        return joblib.load(model_path)

@st.cache_data()
def load_data():
    return pd.read_csv(data_path, delimiter=';')

@st.cache_data()
def load_animation_m():
    return load_lottie_file(machine_json_path)

@st.cache_data()
def load_animation_h():
    return load_lottie_file(hello_json_path)

# Load animations, model and data
lottie_animation_hello = load_animation_h()
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
# model = joblib.load(model_path)
# data = pd.read_csv(data_path)

# Page selection using sidebar
st.sidebar.title("Page Navigator")
page = st.sidebar.selectbox("Choose a page", ["Home", "Data Exploration","Data Visualization", "Data Preprocessing", "Machine Learning Prediction Model","Feedback"])

if page == "Home":
   st.title("Welcome to the Bank Marketing Campaign Prediction App")
   col1, col2 = st.columns(2)
   with col1:
    st.write("""
            This application allows you to predict the success of a bank marketing campaign 
            based on various features. Use the sidebar to navigate to the prediction page 
            and input the required features to get a prediction.
        """)
    with col2:
        st_lottie(lottie_animation_hello, height=300, key="coding")

elif page == "Data Exploration":
    st.title("Let's Explore Our Dataset")
    st.write("The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).")
    st.write("Additional Information:")
    st.write(" The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.")
    st.write("There are four datasets which can be used but our work based on bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.")
    st.write("Luckily We don't have any missing values in our dataset so it seems there is no need to data imputation.")
    
elif page == "Data Visualization":
    st.title("Data Visualization")
    st.write("In this page you can visualise the general information of our dataset.")
    num_rows = st.number_input("Select number of rows to view", min_value=5, max_value=50, value=10)
     # Show the first few rows of the data set
    st.write(f"Here is a preview of the first {num_rows} rows of the dataset:")
    st.write(data.head(num_rows))
    st.write(feature_info)
    st.subheader("Null Values in Each Column")
    null_values = data.isnull().sum()
    
    # Display null values as a table
    st.write(null_values)
    
    # Check if there are any duplicates
    st.subheader("Duplicates")
    duplicates = data.duplicated()
    st.write(f"Number of duplicate rows: {duplicates.sum()}")

    st.subheader("Age Distribution Graph")
    fig, ax = plt.subplots()
    sns.histplot(data['age'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Job Distribution Graph")
    fig, ax = plt.subplots()
    sns.countplot(data['job'], ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Marital Status Distribution Graph")
    fig, ax = plt.subplots()
    sns.countplot(data['marital'], ax=ax)
    st.pyplot(fig)
    
    st.subheader("Correlation Graph")
    fig, ax = plt.subplots(figsize=(12, 8))  
    numerical_data = data.select_dtypes(include=['float64', 'int64']) 
    corr = numerical_data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
elif page == "Data Preprocessing":
    st.title("Data Preprocessing")
    st.write("On this page we will do the appropriate data preprocessing to make our model's life easier.")
    st.subheader("Conversion of numerical columns")
    st.write("When working with numerical operations, it's essential that the data is in a numerical format. If a column intended to be numeric contains text or other non-numeric characters, it can cause errors or incorrect calculations.")
    st.subheader("Binary Encoding")
    st.write("We encode binaric categories as 0 for not or 1 for true.")

            # Show the processed data
    st.subheader("Processed Data")
    st.dataframe(data.dtypes)     
    # Convert numeric columns to appropriate types
    numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert invalid values to NaN if needed

    # Convert categorical columns to 'category' type
    categorical_columns = ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']
    for col in categorical_columns:
        data[col] = data[col].astype('category')

    # Create a dictionary for binary encoding
    binary_dict = {'yes': 1, 'no': 0}

    # Apply the dictionary mapping to your binary columns
    binary_columns = ['default', 'housing', 'loan', 'y']
    for col in binary_columns:
        data[col] = data[col].apply(lambda x: binary_dict.get(x, x))  # using .get to handle missing values safely

    # Apply the dictionary mapping to your binary columns
    binary_dict = {'yes': 1, 'no': 0}
    data['default'] = data['default'].map(binary_dict).fillna(0)  # Default to 0 for unknown values
    data['housing'] = data['housing'].map(binary_dict).fillna(0)
    data['loan'] = data['loan'].map(binary_dict).fillna(0)        

   
    
    # st.write("""
    #     Welcome to the 'Data Visualization' page. This page is dedicated to the Bank Marketing Data Set used in training our model. Here, you can explore the dataset and examine all the features and data within it. Furthermore, you can also explore various graphical representations of these features.
    # """)
    # num_rows = st.number_input("Select number of rows to view", min_value=5, max_value=50, value=10)
    #  # Show the first few rows of the data set
    # st.write(f"Here is a preview of the first {num_rows} rows of the dataset:")
    # st.write(data.head(num_rows))
    # st.write(feature_info)

    # # Data Visualization part for our dataset
    # st.write("### Distribution of Age")
    # fig, ax = plt.subplots()
    # sns.histplot(data['age'], bins=30, kde=True, ax=ax)
    # st.pyplot(fig)

    # st.write("### Job Distribution")
    # fig, ax = plt.subplots()
    # sns.countplot(data['job'], ax=ax)
    # plt.xticks(rotation=45)
    # st.pyplot(fig)

    # st.write("### Marital Status Distribution")
    # fig, ax = plt.subplots()
    # sns.countplot(data['marital'], ax=ax)
    # st.pyplot(fig)

    # st.write("### Education Level Distribution")
    # fig, ax = plt.subplots()
    # sns.countplot(data['education'], ax=ax)
    # plt.xticks(rotation=45)
    # st.pyplot(fig)

    # st.write("### Correlation Heatmap")
    # fig, ax = plt.subplots(figsize=(12, 8))  
    # numerical_data = data.select_dtypes(include=['float64', 'int64']) 
    # corr = numerical_data.corr()
    # sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    # st.pyplot(fig)

    # st.write("### Duration vs Age")
    # fig, ax = plt.subplots()
    # sns.scatterplot(data=data, x='age', y='duration', ax=ax)
    # st.pyplot(fig)

    # st.write("### Box Plot of Campaign Outcome by Job Type")
    # fig, ax = plt.subplots()
    # sns.boxplot(data=data, x='job', y='duration', ax=ax)
    # plt.xticks(rotation=45)
    # st.pyplot(fig)
    
