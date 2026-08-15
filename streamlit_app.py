import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Linear Regression App",
    page_icon="📊",
    layout="wide"
)

# ============================================
# CUSTOM CSS - 3D GLASSMORPHISM STYLE
# ============================================
st.markdown("""
<style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background - Deep Space Gradient */
    .stApp {
        background: radial-gradient(ellipse at 50% 0%, #1a1f3a 0%, #0b0e1a 100%);
    }
    
    /* ===== 3D GLASS CARD ===== */
    .glass-3d {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: 28px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 28px 32px;
        box-shadow: 
            0 30px 60px -15px rgba(0, 0, 0, 0.8),
            inset 0 1px 0 rgba(255, 255, 255, 0.06),
            0 0 80px rgba(79, 172, 254, 0.03);
        transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    
    .glass-3d::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.03) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .glass-3d:hover {
        transform: translateY(-6px) scale(1.003);
        box-shadow: 
            0 40px 80px -20px rgba(0, 0, 0, 0.9),
            inset 0 1px 0 rgba(255, 255, 255, 0.08),
            0 0 120px rgba(79, 172, 254, 0.06);
        border-color: rgba(255, 255, 255, 0.12);
    }
    
    /* ===== 3D TITLE ===== */
    .title-3d {
        font-size: 3.8rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 40%, #4facfe 80%, #00f2fe 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 6s ease-in-out infinite;
        text-shadow: 
            0 0 60px rgba(245, 87, 108, 0.2),
            0 20px 40px rgba(0, 0, 0, 0.4);
        letter-spacing: -2px;
        transform: perspective(1200px) rotateX(3deg);
        margin-bottom: 0 !important;
        line-height: 1.1;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .subtitle-3d {
        font-size: 1rem !important;
        font-weight: 300 !important;
        color: rgba(255, 255, 255, 0.35) !important;
        letter-spacing: 6px;
        text-transform: uppercase;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        margin-top: -4px !important;
    }
    
    /* ===== 3D METRIC CARDS ===== */
    .metric-3d {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 20px 24px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 
            0 20px 40px -12px rgba(0, 0, 0, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .metric-3d::after {
        content: '';
        position: absolute;
        top: -100%;
        left: -100%;
        width: 300%;
        height: 300%;
        background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.03) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .metric-3d:hover {
        transform: translateY(-8px) scale(1.03);
        box-shadow: 
            0 30px 60px -15px rgba(0, 0, 0, 0.8),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.12);
    }
    
    .metric-value {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #f093fb 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
        letter-spacing: -1px;
    }
    
    .metric-label {
        font-size: 0.75rem !important;
        color: rgba(255, 255, 255, 0.35) !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 400;
        margin-top: 6px;
    }
    
    /* ===== SIDEBAR ===== */
    .css-1d391kg, .css-12oz5g7 {
        background: rgba(11, 14, 26, 0.85) !important;
        backdrop-filter: blur(30px) !important;
        -webkit-backdrop-filter: blur(30px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    
    .css-1d391kg .stSelectbox, .css-12oz5g7 .stSelectbox {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    
    /* ===== BUTTONS - 3D GLOW ===== */
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        border: none !important;
        border-radius: 18px !important;
        padding: 14px 36px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        color: #0b0e1a !important;
        box-shadow: 
            0 12px 40px -8px rgba(79, 172, 254, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        letter-spacing: 0.5px;
        transform: perspective(800px) rotateX(2deg);
        text-transform: uppercase;
        font-size: 0.85rem !important;
    }
    
    .stButton > button:hover {
        transform: perspective(800px) rotateX(0deg) translateY(-4px) scale(1.02) !important;
        box-shadow: 
            0 20px 50px -8px rgba(79, 172, 254, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }
    
    .stButton > button:active {
        transform: scale(0.97) !important;
    }
    
    /* ===== HEADERS ===== */
    h1, h2, h3, h4, h5, h6 {
        color: rgba(255, 255, 255, 0.85) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        font-size: 1.8rem !important;
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.5));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 16px !important;
    }
    
    /* ===== DATAFRAME ===== */
    .stDataFrame {
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }
    
    .stDataFrame thead tr th {
        background: rgba(255, 255, 255, 0.05) !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-weight: 600 !important;
        padding: 12px !important;
    }
    
    .stDataFrame tbody tr td {
        color: rgba(255, 255, 255, 0.6) !important;
        padding: 10px !important;
    }
    
    .stDataFrame tbody tr:hover {
        background: rgba(255, 255, 255, 0.03) !important;
    }
    
    /* ===== INPUTS & SELECTS ===== */
    .stSelectbox, .stMultiSelect, .stSlider, .stNumberInput {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label {
        color: rgba(255, 255, 255, 0.5) !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    
    /* ===== CHECKBOX ===== */
    .stCheckbox label {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    .stCheckbox label span {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* ===== ALERTS ===== */
    .stAlert {
        border-radius: 16px !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    
    .stAlert .stAlertContent {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* ===== MATPLOTLIB FIGURES ===== */
    .stImage {
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        background: rgba(255, 255, 255, 0.02) !important;
        padding: 4px !important;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        color: rgba(255, 255, 255, 0.6) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    /* ===== SUCCESS / WARNING / INFO ===== */
    .stSuccess, .stWarning, .stInfo, .stError {
        border-radius: 16px !important;
        backdrop-filter: blur(12px) !important;
    }
    
    /* ===== PREDICTION OUTPUT ===== */
    .prediction-box {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.12), rgba(240, 147, 251, 0.12));
        border-radius: 20px;
        padding: 28px 32px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.4);
        margin-top: 16px;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 20%, rgba(79,172,254,0.05) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .prediction-value {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #f093fb, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -2px;
        line-height: 1.2;
    }
    
    .prediction-label {
        font-size: 0.75rem !important;
        color: rgba(255, 255, 255, 0.3) !important;
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-bottom: 4px;
    }
    
    .prediction-range {
        font-size: 0.85rem !important;
        color: rgba(255, 255, 255, 0.25) !important;
        margin-top: 8px;
    }
    
    /* ===== EQUATION BOX ===== */
    .equation-box {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    
    .equation-text {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.85);
        letter-spacing: 0.3px;
        font-weight: 500;
        font-family: 'Inter', monospace;
    }
    
    .equation-sub {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.2);
        margin-top: 8px;
        letter-spacing: 1px;
    }
    
    /* ===== FOOTER ===== */
    .footer-3d {
        text-align: center;
        padding: 30px 0 10px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        color: rgba(255, 255, 255, 0.1);
        font-size: 0.75rem;
        letter-spacing: 3px;
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .title-3d {
            font-size: 2.4rem !important;
        }
        .glass-3d {
            padding: 16px 18px !important;
        }
        .metric-value {
            font-size: 1.8rem !important;
        }
        .prediction-value {
            font-size: 2.4rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown('<p class="title-3d">📊 Linear Regression Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-3d">Machine Learning With Sajib • 3D Premium Edition</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### 🚀 Control Panel")
    st.markdown("---")
    
    use_ex = st.checkbox("✨ Use Example Dataset (Tips)", value=True)
    
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
        st.success("✅ Loaded: 'tips' dataset")
    else:
        upload_file = st.file_uploader(
            "📁 Upload CSV",
            type=["csv"],
            help="Upload a CSV with at least two numeric columns"
        )
        if upload_file is not None:
            df = load_uploaded_data(upload_file)
            st.success("✅ File uploaded!")
        else:
            st.warning("⚠️ Please upload a CSV or use example dataset.")
            st.stop()
    
    st.markdown("---")
    
    # Dataset Info
    st.markdown("### 📊 Dataset Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📋 Rows", df.shape[0])
    with col2:
        st.metric("📐 Cols", df.shape[1])
    
    st.caption(f"🔢 Numeric: {len(df.select_dtypes(include=np.number).columns)}")
    st.markdown("---")
    st.caption("⚡ Built with Streamlit • 3D Edition")

# ============================================
# MAIN CONTENT - GLASS CARDS
# ============================================

# --- Dataset Preview ---
st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
st.subheader("📄 Dataset Preview")
st.dataframe(df, use_container_width=True)

if st.checkbox("📊 Show Dataset Statistics"):
    st.dataframe(df.describe(), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Model Configuration ---
st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
st.subheader("⚙️ Model Configuration")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("❌ Need at least two numeric columns for regression")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    target = st.selectbox(
        "🎯 Select Target Variable (Dependent)",
        numeric_cols,
        help="The variable you want to predict"
    )

with col2:
    features = st.multiselect(
        "📌 Select Feature Columns (Independent)",
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
    "📊 Test Set Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
    help="Proportion of data to use for testing"
)
st.markdown('</div>', unsafe_allow_html=True)

# --- Train Model ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
with st.spinner("🧠 Training model..."):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

# --- Model Performance ---
st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
st.subheader("📈 Model Performance")

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-3d">
        <div class="metric-value">{mse:.3f}</div>
        <div class="metric-label">📉 MSE</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-3d">
        <div class="metric-value">{rmse:.3f}</div>
        <div class="metric-label">📊 RMSE</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-3d">
        <div class="metric-value">{r2:.3f}</div>
        <div class="metric-label">🎯 R² Score</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(features) - 1) if len(y_test) > len(features) + 1 else r2
    st.markdown(f"""
    <div class="metric-3d">
        <div class="metric-value">{adj_r2:.3f}</div>
        <div class="metric-label">📐 Adj. R²</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Feature Importance ---
st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
st.subheader("📊 Feature Coefficients")

coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})
coef_df['Absolute Coefficient'] = np.abs(coef_df['Coefficient'])
coef_df = coef_df.sort_values('Absolute Coefficient', ascending=True)

# Enhanced Matplotlib Figure
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('none')
ax.set_facecolor('none')

bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], 
               color=plt.cm.viridis(np.linspace(0, 1, len(coef_df))),
               edgecolor='rgba(255,255,255,0.1)',
               linewidth=1,
               height=0.6)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, coef_df['Coefficient'])):
    ax.text(val + (0.02 * abs(val) if val >= 0 else -0.02 * abs(val)), 
            bar.get_y() + bar.get_height()/2, 
            f'{val:.3f}', 
            va='center', 
            ha='left' if val >= 0 else 'right',
            color='rgba(255,255,255,0.5)',
            fontsize=10)

ax.set_xlabel('Coefficient Value', color='rgba(255,255,255,0.5)', fontsize=11)
ax.axvline(x=0, color='rgba(255,255,255,0.2)', linestyle='--', linewidth=1)
ax.tick_params(colors='rgba(255,255,255,0.4)')
ax.spines['bottom'].set_color('rgba(255,255,255,0.05)')
ax.spines['left'].set_color('rgba(255,255,255,0.05)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.label.set_color('rgba(255,255,255,0.4)')
ax.yaxis.label.set_color('rgba(255,255,255,0.4)')
ax.tick_params(axis='both', colors='rgba(255,255,255,0.2)')

plt.tight_layout()
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# --- Actual vs Predicted ---
st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
st.subheader("🎯 Actual vs Predicted Values")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('none')
ax.set_facecolor('none')

scatter = ax.scatter(y_test, y_pred, 
                     alpha=0.7, 
                     c=y_pred - y_test, 
                     cmap='RdBu_r',
                     edgecolors='rgba(255,255,255,0.15)',
                     linewidth=0.5,
                     s=80)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', linewidth=2, alpha=0.6, label='Perfect Prediction')

ax.set_xlabel('Actual Values', color='rgba(255,255,255,0.5)', fontsize=11)
ax.set_ylabel('Predicted Values', color='rgba(255,255,255,0.5)', fontsize=11)
ax.tick_params(colors='rgba(255,255,255,0.3)')
ax.spines['bottom'].set_color('rgba(255,255,255,0.05)')
ax.spines['left'].set_color('rgba(255,255,255,0.05)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.label.set_color('rgba(255,255,255,0.4)')
ax.yaxis.label.set_color('rgba(255,255,255,0.4)')

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Residual', color='rgba(255,255,255,0.4)')
cbar.ax.tick_params(colors='rgba(255,255,255,0.2)')

plt.tight_layout()
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# --- Residual Analysis ---
st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
st.subheader("📊 Residual Analysis")

residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('none')

# Residuals vs Predicted
axes[0].set_facecolor('none')
axes[0].scatter(y_pred, residuals, alpha=0.6, 
                edgecolors='rgba(255,255,255,0.1)',
                linewidth=0.5, s=60,
                c=residuals, cmap='RdBu_r')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6)
axes[0].set_xlabel('Predicted Values', color='rgba(255,255,255,0.4)')
axes[0].set_ylabel('Residuals', color='rgba(255,255,255,0.4)')
axes[0].tick_params(colors='rgba(255,255,255,0.2)')
for spine in axes[0].spines.values():
    spine.set_color('rgba(255,255,255,0.05)')

# Histogram of Residuals
axes[1].set_facecolor('none')
axes[1].hist(residuals, bins=20, edgecolor='rgba(255,255,255,0.2)', 
             color='rgba(79,172,254,0.4)', linewidth=1)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.6)
axes[1].set_xlabel('Residuals', color='rgba(255,255,255,0.4)')
axes[1].set_ylabel('Frequency', color='rgba(255,255,255,0.4)')
axes[1].tick_params(colors='rgba(255,255,255,0.2)')
for spine in axes[1].spines.values():
    spine.set_color('rgba(255,255,255,0.05)')

plt.tight_layout()
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# --- Model Equation ---
st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
st.subheader("📝 Model Equation")

intercept = model.intercept_
coefficients = model.coef_

equation = f"{target} = {intercept:.3f}"
for i, (feature, coef) in enumerate(zip(features, coefficients)):
    if coef >= 0:
        equation += f" + {coef:.3f}·{feature}"
    else:
        equation += f" - {abs(coef):.3f}·{feature}"

st.markdown(f"""
<div class="equation-box">
    <div class="equation-text">{equation}</div>
    <div class="equation-sub">Intercept: {intercept:.3f} • Features: {len(features)}</div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Section ---
st.markdown('<div class="glass-3d">', unsafe_allow_html=True)
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
            format="%.2f",
            key=f"pred_input_{feature}"
        )
        input_values.append(val)

if st.button("🔮 Predict", use_container_width=True):
    input_df = pd.DataFrame([input_values], columns=features)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">Predicted {target}</div>
        <div class="prediction-value">{prediction:.3f}</div>
        <div class="prediction-range">± {rmse:.3f} (RMSE)</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer-3d">
    Built with ❤️ using Streamlit • 3D Linear Regression Pro
</div>
""", unsafe_allow_html=True)
