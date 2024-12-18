import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import json
import os
import numpy as np


current_dir = os.path.dirname(__file__)
hello_json_path = os.path.join(current_dir, 'hello.json')
machine_json_path = os.path.join(current_dir, 'machine.json')
confused_json_path = os.path.join(current_dir, 'confused.json')
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
@st.cache_data()
def load_animation_c():
    return load_lottie_file(confused_json_path)

# Load animations, model and data
lottie_animation_hello = load_animation_h()
lottie_animation_machine = load_animation_m()
lottie_confused_animation = load_animation_c()
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
pipeline = joblib.load('logistic_regression_pipeline.pkl')
# pipelineRF = joblib.load('random_forest_pipeline.pkl')
# Page selection using sidebar
st.sidebar.title("Page Navigator")
page = st.sidebar.selectbox("Choose a page", ["Home", "Data Exploration","Data Visualization", "Data Preprocessing", "Model Training","Logistic Regression Prediction Model"])

if page == "Home":
   st.title("Bank Marketing Campaign Prediction App")
   col1, col2 = st.columns(2)
   with col1:
    st.subheader("Welcome!")
    st.write("This app showcases the process of a machine learning model which predicts of outcomes of marketing campaign of via phone calls of a Portuguese bank institution.")
    st.write("You can navigate between pages using the sidebar and follow the process of model training.")
    with col2:
        st_lottie(lottie_animation_machine, height=300, key="coding")

elif page == "Data Exploration":
    st.title("Let's Explore Our Dataset")
    st.write("The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit")
    st.write("Additional Information:")
    st.write(" The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.")
    st.write("There are four datasets which can be used but our work based on bank-additional.csv with 10% of the examples (4119), randomly selected from 1, and 20 inputs.")
    st.write("Luckily We don't have any missing values in our dataset so it seems there is no need to data imputation.")
    
elif page == "Data Visualization":
    st.title("Data Visualization")
    st.write("In this page you can visualise the general information of our dataset.")
     # Show the first few rows of the data set
    st.write("Here is a preview of the first 10 rows of the dataset:")
    st.write(data.head(10))
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
    st.write("We have encoded binaric categories as 0 for false or 1 for true.")

    st.subheader("Label Encoding")
    st.write("After that we label encoded the categorical columns into numerical ones so that they can be fitted by machine learning models which only take numerical data.")

    st.subheader("One-Hot Encoding For the Nominal Categorical Variables")
    st.write("With that we eliminated ordinality between the variables.")

    st.subheader("Outliers")
    st.write("When we analyze the pairplot of variables we saw few outliers but we decided to don't remove them in order to they carry important information of sample behaviour.")

    st.subheader("Scaling")
    st.write("We have scaled our data set with by standart scaling method ")     
    st.write("StandardScaler operates on the principle of normalization, where it transforms the distribution of each feature to have a mean of zero and a standard deviation of one.")
    st.write("This process ensures that all features are on the same scale, preventing any single feature from dominating the learning process due to its larger magnitude.")    

    # Show the processed data
    st.subheader("Processed Data")
    st.dataframe(data.dtypes)
    st.write(data.head())

elif page == "Model Training":
    st.title("Model Training")
    st.write("Our data is ready to serve our ML model. Let's move on to model training phase")

    st.subheader("Data Splitting")
    st.write("In order to prevent over fitting we split the data as training and test data by 8/2 ratio")

    st.subheader("Plain Models")
    st.write("First we tested how good plain machine learning models.")
    st.write("Logistic regression model achieved %92 accuracy.")
    st.write("Random forest model achieved %90 accuracy.")
    st.write("SVM model achieved %91 accuracy.")
    st.write("From this point we decided to move on with Logistic regression model")

    st.subheader("Grid Search with Cross Validation")
    st.write("We used grid search technique with cross validation and found the best hyperparameters as:")
    st.write("Best Hyperparameters: {'C': 1, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'}")


    st.subheader("Feature Selection")
    st.write("We search the important features with KBest Anova F-test to decide which features have over or under weight which affects our model's training and it increased the accuracy slightly.")
    st.write("The algorithm found that the variables ['duration', 'pdays', 'emp.var.rate', 'euribor3m', 'nr.employed'] are the best 5 features.")
    st.write("Changing the k value of Kbest function doesn't change the generalization. It just changed the accuracy a very low rate")
    st.write("So we decided to choose these variables to re-train our model again.")

    st.subheader("Re-training with found hyperparameters and L1 penalty")
    st.write("So our final model had approximately %90 accuracy and we saved it.")

    st.subheader("Pipelining")
    st.write("We've created a pipeline as following;")
    
    code = '''# Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Conversion steps for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combining numeric and categorical transformations using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    pipeline = {
        'Logistic Regression': ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('selector', SelectKBest(score_func=f_classif, k=5)),
            ('classifier', LogisticRegression(random_state=42))
        ])
    }'''
    st.code(code, language="python")

    st.write("But this pipeline cost us %5 loss of accuracy.")
    st_lottie(lottie_confused_animation, height=300, width = 700, key="coding")

elif page == "Logistic Regression Prediction Model":
    st.title("Prediction")
    st.write("Now our model is ready to predict possible outcomes!")
    st.subheader("Try giving it some variables from below and see what you got.")
    # Input fields for user to enter feature values
    duration = st.number_input("Duration of Last Contact (seconds)", min_value=0, value=0)
    pdays = st.number_input("Days Since Last Contact", min_value=-1, value=0)
    emp_var_rate = st.number_input("Employment Variation Rate", value=0.0)
    euribor3m = st.number_input("Euribor 3 Month Rate", value=0.0)
    nr_employed = st.number_input("Number of Employees", value=0.0)
    poutcome = st.selectbox("Outcome of Previous Marketing Campaign", ["failure", "nonexistent", "success"])
    cons_price_idx = st.number_input("Consumer Price Index", value=0.0)
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    campaign_previous_interaction = st.number_input("Campaign Previous Interaction", min_value=0, value=0)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=0.0)
    previous = st.number_input("number of contacts performed before this campaign and for this client", min_value = 0, value = 0)
    campaign = st.number_input("number of contacts performed during this campaign and for this client", min_value= 0, value = 0)
    
    # Create dictionary with user inputs
    input_data = {
        'duration': duration,
        'pdays': pdays,
        'emp.var.rate': emp_var_rate,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed,
        'poutcome': poutcome,  
        'cons.price.idx': cons_price_idx,
        'age': age,
        'campaign.previous.interaction': campaign_previous_interaction,
        'cons.conf.idx': cons_conf_idx,
        'previous' : previous,
        'campaign' : campaign
    }
    
    input_df = pd.DataFrame([input_data])

    if st.button("Fingers Crossed..."):
        # Make a prediction using the pipeline
        prediction = pipeline.predict(input_df)
        st.write("Prediction:", prediction[0])