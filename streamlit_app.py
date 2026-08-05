import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklern.model_selection import train_test_split

st.title("Linear Regression Web App")
st.subheader("Machine Learning With Sajib")


st.sidebar.header("Upload CSV data")

use_ex = st.sidebar.checkbox("use Expal Dataset")

#Load
if use_ex:
  df = sns.load_dataset("tips")
  df = df.dropna()
  st.success("Loaded Dataste: 'tips' ")
else: upload_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
 if upload_file:
   df = pd.read_CSV(upload_file)
 else:
    st.warning("plz upload csv file")
    st.stop()
  
 
  
