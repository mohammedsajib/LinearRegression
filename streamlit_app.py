import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Linear Regression App",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Linear Regression Web App")
st.subheader("Machine Learning With Sajib")
st.markdown("---")

# Sidebar
st.sidebar.header("📁 Upload CSV Data")

use_ex = st.sidebar.checkbox("Use Example Dataset (Tips)", value=True)

# Load Dataset
@st.cache_data
def load_example_data():
    df = sns.load_dataset("tips")
    return df.dropna()

@st.cache_data
def load_uploaded_data(file):
    return pd.read_csv(file)

if use_ex:
    df = load_example_data()
    st.sidebar.success("✅ Loaded Dataset: 'tips'")
else:
    upload_file = st.sidebar.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Upload a CSV file with at least two numeric columns"
    )

    if upload_file is not None:
        df = load_uploaded_data(upload_file)
        st.sidebar.success("✅ File uploaded successfully!")
    else:
        st.sidebar.warning("⚠️ Please upload a CSV file or use example dataset.")
        st.stop()

# Display dataset information
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Total Rows", df.shape[0])
with col2:
    st.metric("📋 Total Columns", df.shape[1])
with col3:
    st.metric("🔢 Numeric Columns", len(df.select_dtypes(include=np.number).columns))

# Show Dataset
st.subheader("📄 Dataset Preview")
st.dataframe(df, use_container_width=True)

# Display dataset statistics
if st.checkbox("Show Dataset Statistics"):
    st.write(df.describe())

# Model Configuration
st.markdown("---")
st.subheader("⚙️ Model Configuration")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("❌ Need at least two numeric columns for regression")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    target = st.selectbox(
        "🎯 Select Target Variable (Dependent Variable)",
        numeric_cols,
        help="The variable you want to predict"
    )

with col2:
    features = st.multiselect(
        "📌 Select Feature Columns (Independent Variables)",
        [col for col in numeric_cols if col != target],
        default=[col for col in numeric_cols if col != target],
        help="Variables used to predict the target"
    )

if len(features) == 0:
    st.warning("⚠️ Please select at least one feature")
    st.stop()

# Data preprocessing
df_model = df[features + [target]].dropna()

X = df_model[features]
y = df_model[target]

# Train-Test Split
test_size = st.slider(
    "Test Set Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
    help="Proportion of data to use for testing"
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
with st.spinner("Training model..."):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

# Model Evaluation
st.markdown("---")
st.subheader("📈 Model Performance")

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📉 Mean Squared Error", f"{mse:.3f}")
with col2:
    st.metric("📈 R² Score", f"{r2:.3f}")
with col3:
    st.metric("📊 RMSE", f"{np.sqrt(mse):.3f}")

# Feature Importance/ Coefficients
st.subheader("📊 Feature Coefficients")
coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
coef_df['Absolute Coefficient'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Absolute Coefficient', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'])
ax.set_xlabel('Coefficient Value')
ax.set_title('Feature Coefficients')
ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
st.pyplot(fig)

# Actual vs Predicted Plot
st.subheader("🎯 Actual vs Predicted Values")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.set_title(f'Actual vs Predicted (R² = {r2:.3f})')
plt.tight_layout()
st.pyplot(fig)

# Residuals Plot
st.subheader("📊 Residual Analysis")

residuals = y_test - y_pred
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Residuals vs Predicted
axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Predicted')

# Histogram of Residuals
axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')

plt.tight_layout()
st.pyplot(fig)

# Model Equation
st.subheader("📝 Model Equation")
intercept = model.intercept_
coefficients = model.coef_

equation = f"**{target} = {intercept:.3f}"
for i, (feature, coef) in enumerate(zip(features, coefficients)):
    if coef >= 0:
        equation += f" + {coef:.3f} * {feature}"
    else:
        equation += f" - {abs(coef):.3f} * {feature}"
equation += "**"

st.write(equation)

# Prediction Section
st.markdown("---")
st.subheader("🔮 Make Predictions")

st.write("Enter values for the features to predict the target:")

input_values = []
cols = st.columns(min(len(features), 4))
for i, feature in enumerate(features):
    with cols[i % 4]:
        val = st.number_input(
            f"{feature}",
            value=float(X[feature].mean()),
            step=0.1,
            format="%.2f"
        )
        input_values.append(val)

if st.button("🔮 Predict", use_container_width=True):
    input_df = pd.DataFrame([input_values], columns=features)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"**Predicted {target}:** {prediction:.3f}")
    
    # Show prediction interval (approximate)
    st.info(f"📊 The predicted value is approximately {prediction:.3f} ± {np.sqrt(mse):.3f}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Linear Regression Model")
