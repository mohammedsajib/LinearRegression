import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
target = st.selectbox("select Target output", numeric_cols)
f = st.multiselect("select input featur colums", [col for col in numeric_cols if col = target])



