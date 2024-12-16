#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import streamlit as st
import pickle


# In[2]:


# Read the dataset with the correct delimiter
df = pd.read_csv(r'C:\Users\Halil\Desktop\projectcsv\bank-additional.csv', delimiter=';')
df.head()


# In[3]:


# statistical info
df.describe()


# In[4]:


# datatype info
df.info()


# In[5]:


# check for null values
df.isnull().sum()


# In[6]:


# Check if there are any duplicates
duplicates = df.duplicated()

# Display the number of duplicates
print(f"Number of duplicate rows: {duplicates.sum()}")


# In[8]:


# Convert numeric columns to appropriate types
numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                   'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert invalid values to NaN if needed


# In[9]:


# Convert categorical columns to 'category' type
categorical_columns = ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']
for col in categorical_columns:
    df[col] = df[col].astype('category')


# In[10]:


# Create a dictionary for binary encoding
binary_dict = {'yes': 1, 'no': 0}

# Apply the dictionary mapping to your binary columns
binary_columns = ['default', 'housing', 'loan', 'y']
for col in binary_columns:
    df[col] = df[col].apply(lambda x: binary_dict.get(x, x))  # using .get to handle missing values safely


# In[11]:


# Apply the dictionary mapping to your binary columns
binary_dict = {'yes': 1, 'no': 0}
df['default'] = df['default'].map(binary_dict).fillna(0)  # Default to 0 for unknown values
df['housing'] = df['housing'].map(binary_dict).fillna(0)
df['loan'] = df['loan'].map(binary_dict).fillna(0)


# In[12]:


# Check the updated data types
print(df.dtypes)


# In[13]:


# List of columns to be label encoded (Ordinal Categorical)
label_encoded_cols = ['education']  # Add more ordinal columns if needed

# Create LabelEncoder instance
le = LabelEncoder()

# Apply LabelEncoder to each of the columns
for col in label_encoded_cols:
    df[col] = le.fit_transform(df[col])


# In[14]:


# List of columns to apply One-Hot Encoding (Nominal Categorical)
one_hot_encoded_cols = ['job', 'marital', 'contact', 'month', 'day_of_week', 'poutcome']  # Example columns

# Apply One-Hot Encoding using pd.get_dummies
df = pd.get_dummies(df, columns=one_hot_encoded_cols, drop_first=True)


# In[15]:


print(df.head())


# In[19]:


# Define numeric columns for outlier detection
numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                   'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Calculate IQR for each numeric column
Q1 = df[numeric_columns].quantile(0.25)
Q3 = df[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

# Define outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Check the calculated bounds
print(lower_bound)
print(upper_bound)


# In[20]:


corr = df.corr()
plt.figure(figsize=(21,21))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[21]:


# Pairplot for multiple numeric columns
numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                   'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                   'euribor3m', 'nr.employed']
sns.pairplot(df[numeric_columns])
plt.show()

#it seems to me we don't need to exclude outliers they could carry information i'm not sure.


# In[26]:


from sklearn.preprocessing import StandardScaler

# Numeric columns
numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                   'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Initialize the scaler
scaler = StandardScaler()

# Scale the numeric columns
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


# In[27]:


# Define the feature columns (X) and the target column (y)
X = df.drop('y', axis=1)  # Drop the target column
y = df['y']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[28]:


# Initialize the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=500)

# Train the model
model.fit(X_train, y_train)


# In[29]:


# Make predictions
y_pred = model.predict(X_test)


# In[30]:


# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")


# In[31]:


# Get feature importance
importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
importance = importance.sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance:")
print(importance)


# In[33]:


# Get feature importance (coefficients)
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

# Sort by the absolute value of coefficients for better visualization
importance['AbsCoefficient'] = importance['Coefficient'].abs()
importance = importance.sort_values(by='AbsCoefficient', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=importance, palette='viridis')

# Add labels and title
plt.title('Feature Importance (Logistic Regression Coefficients)', fontsize=14)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()


# In[34]:


# Split the data into features (X) and target (y)
X = df.drop(columns=['y'])  # Drop the target variable
y = df['y']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("\nAccuracy:", accuracy)


# In[35]:


# Get feature importances
feature_importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()


# In[36]:


# Split the data into features (X) and target (y)
X = df.drop(columns=['y'])  # Drop the target variable
y = df['y']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Support Vector Machine model
svm_model = SVC(kernel='linear', random_state=42)  # Using a linear kernel for simplicity

# Fit the model on the training data
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display results
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("\nAccuracy:", accuracy)


# In[ ]:


# Define the model
logreg = LogisticRegression(random_state=42)

# Define the hyperparameters to tune
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],  # Try both L1 and L2 penalties
    'solver': ['liblinear', 'lbfgs', 'saga'],  # 'liblinear' supports 'l1', 'lbfgs' supports 'l2'
    'max_iter': [100, 200, 500]
}
# Set up GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, 
                           cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Display the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model from the grid search
best_logreg = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_logreg.predict(X_test)
print("\nAccuracy on Test Set:", accuracy_score(y_test, y_pred))


# In[ ]:


logreg = LogisticRegression(C=1, max_iter=100, penalty='l2', solver='liblinear')

# Train the model
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[37]:


# Split the data into features and target
X = df[numeric_columns]
y = df['y']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SelectKBest with ANOVA F-test to select top 5 features
selector = SelectKBest(score_func=f_classif, k=5)  # Select top 5 features
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)

# Check which features were selected
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {selected_features}")


# In[38]:


# Split the data into features and target
X = df[numeric_columns]
y = df['y']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SelectKBest with ANOVA F-test to select top 5 features
selector = SelectKBest(score_func=f_classif, k=5)  # Select top 5 features
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)

# Check which features were selected
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {selected_features}")

# Train a logistic regression model on the selected features
logreg = LogisticRegression(C=1, max_iter=100, penalty='l1', solver='liblinear')
logreg.fit(X_train_new, y_train)

# Evaluate the model
y_pred = logreg.predict(X_test_new)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set with selected features: {accuracy}")


# In[39]:


# Split the data into features and target
X = df[numeric_columns]
y = df['y']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model with L1 regularization
logreg = LogisticRegression(C=1, max_iter=100, penalty='l1', solver='liblinear')
logreg.fit(X_train, y_train)

# Get the coefficients (weights) of the features
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logreg.coef_.flatten()
})

# Sort features by the absolute value of the coefficients
coefficients['abs_coefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='abs_coefficient', ascending=False)

# Display the coefficients and their absolute values
print("Coefficients and their absolute values:\n", coefficients)

# Select features with non-zero coefficients
selected_features = coefficients[coefficients['Coefficient'] != 0]['Feature']
print(f"Selected features (non-zero coefficients): {selected_features}")

# Train a Logistic Regression model on the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
logreg_selected = LogisticRegression(C=1, max_iter=100, penalty='l1', solver='liblinear')
logreg_selected.fit(X_train_selected, y_train)

# Evaluate the model on the test set
y_pred = logreg_selected.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set with selected features: {accuracy}")


# In[40]:


# Select the numeric columns for polynomial feature generation
numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                   'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Initialize PolynomialFeatures (degree=2 for quadratic features)
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

# Generate polynomial features (this will include both the original features and interaction terms)
poly_features = poly.fit_transform(df[numeric_columns])

# Create a DataFrame with the new polynomial features
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numeric_columns))

# Concatenate the polynomial features to the original DataFrame
df = pd.concat([df, poly_df], axis=1)

# Optionally, check the first few rows to ensure the new features are added
print(df.head())


# In[54]:


# Step 1: Define the features (X) and the target (y)
X = df.drop(columns=['y'])
y = df['y']

# Step 2: Split the data into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize and train the Logistic Regression model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1')  # L1 regularization
logreg.fit(X_train, y_train)

# Step 4: Evaluate the model on the test set
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on the test set
y_pred = logreg.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on Test Set: {accuracy}")

# Classification report (precision, recall, f1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[51]:


# Split the dataset into features and target
X = df.drop(columns=['y'])  # Features (exclude target 'y')
y = df['y']  # Target (y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForest model
rf_model = RandomForestClassifier(random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy on Test Set: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


# In[52]:


# streamlit
model_filename = "logistic_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(logreg, file)

st.title("Logistic Regression Model for Bank Marketing")
st.write("Train and download the logistic regression model")
st.write("Training Accuracy:", logreg.score(X_train, y_train))
st.write("Testing Accuracy:", logreg.score(X_test, y_test))
with open(model_filename, "rb") as file:
    st.download_button(
        label="Download Logistic Regression Model",
        data=file,
        file_name="logistic_model.pkl",
        mime="application/octet-stream"
    )


# In[53]:


# Split the dataset into training and test sets
X = df.drop(columns=['y'])  # assuming 'y' is the target column
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model with balanced class weights
model = LogisticRegression(class_weight='balanced', max_iter=100, solver='liblinear')

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[106]:


# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit SMOTE on the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train the logistic regression model with resampled data
model = LogisticRegression(max_iter=100, solver='liblinear')
model.fit(X_train_resampled, y_train_resampled)

# Predictions on the original test set
y_pred_resampled = model.predict(X_test)

# Evaluate performance
accuracy_resampled = accuracy_score(y_test, y_pred_resampled)
print("Accuracy on Test Set after SMOTE:", accuracy_resampled)

print("\nClassification Report after SMOTE:")
print(classification_report(y_test, y_pred_resampled))

print("\nConfusion Matrix after SMOTE:")
print(confusion_matrix(y_test, y_pred_resampled))


# In[ ]:




