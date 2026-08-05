import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
