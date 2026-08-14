import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.title("Linear Regression Web App")
st.subheader("Machine Learning With Sajib")

# Sidebar
st.sidebar.header("Upload CSV Data")

use_ex = st.sidebar.checkbox("Use Example Dataset")

# Load Dataset
if use_ex:
    df = sns.load_dataset("tips")
    df = df.dropna()
    st.success("Loaded Dataset: 'tips'")

else:
    upload_file = st.sidebar.file_uploader(
        "Upload your CSV file",
        type=["csv"]
    )

    if upload_file is not None:
        df = pd.read_csv(upload_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Show Dataset
st.subheader("Dataset Preview")
st.dataframe(df)

#model
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least two numeric columns")
    st.stop()
target = st.selectbox("Select Target Output", numeric_cols)

features = st.multiselect(
    "Select Input Feature Columns",
    [col for col in numeric_cols if col != target],
     default= [col for col in numeric_cols if col != target]
)

if len(features) == 0:
    st.write("Please select at least one feature")
    st.stop()

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)








