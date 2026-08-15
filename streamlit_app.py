import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="3D Linear Regression Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CUSTOM CSS - 3D / GLASSMORPHISM STYLE
# ========================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with subtle gradient */
    .stApp {
        background: linear-gradient(145deg, #0b0e1a 0%, #1a1f35 100%);
    }
    
    /* Glass card effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8), inset 0 1px 2px rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        margin-bottom: 20px;
    }
    
    .glass-card:hover {
        transform: translateY(-4px) scale(1.005);
        box-shadow: 0 35px 60px -15px rgba(0, 0, 0, 0.9), inset 0 1px 2px rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* 3D Title */
    .title-3d {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(245, 87, 108, 0.3), 0 8px 32px rgba(0,0,0,0.5);
        letter-spacing: -1px;
        margin-bottom: 0 !important;
        transform: perspective(1000px) rotateX(2deg);
    }
    
    .subtitle-3d {
        font-size: 1.1rem !important;
        font-weight: 300 !important;
        color: rgba(255, 255, 255, 0.5) !important;
        letter-spacing: 4px;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 16px;
    }
    
    /* Metric cards with 3D depth */
    .metric-3d {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 20px 24px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .metric-3d:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 30px 60px -15px rgba(0, 0, 0, 0.8);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .metric-value {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #f093fb 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.8rem !important;
        color: rgba(255, 255, 255, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 400;
        margin-top: 4px;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background: rgba(11, 14, 26, 0.8) !important;
        backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Buttons with 3D effect */
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        color: #0b0e1a !important;
        box-shadow: 0 12px 30px -8px rgba(79, 172, 254, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        letter-spacing: 0.5px;
        transform: perspective(600px) rotateX(2deg);
    }
    
    .stButton > button:hover {
        transform: perspective(600px) rotateX(0deg) translateY(-3px) scale(1.02) !important;
        box-shadow: 0 20px 40px -8px rgba(79, 172, 254, 0.6) !important;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox, .stMultiSelect, .stSlider, .stNumberInput {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Success/Warning/Error */
    .stAlert {
        border-radius: 16px !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Fix plotly container */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 16px !important;
        background: rgba(255, 255, 255, 0.02) !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.25);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 6px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 8px 20px !important;
        color: rgba(255, 255, 255, 0.5) !important;
        font-weight: 500 !important;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(79, 172, 254, 0.2), rgba(240, 147, 251, 0.2)) !important;
        color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# HEADER WITH 3D EFFECT
# ========================
st.markdown('<p class="title-3d">📊 3D Linear Regression Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-3d">Machine Learning With Sajib • Advanced Analytics Suite</p>', unsafe_allow_html=True)

# ========================
# SIDEBAR
# ========================
with st.sidebar:
    st.markdown("### 🚀 Control Panel")
    st.markdown("---")
    
    use_ex = st.checkbox("✨ Use Example Dataset (Tips)", value=True)
    
    if use_ex:
        @st.cache_data
        def load_example_data():
            df = sns.load_dataset("tips")
            return df.dropna()
        df = load_example_data()
        st.success("✅ Loaded: `tips` dataset")
    else:
        upload_file = st.file_uploader(
            "📁 Upload CSV",
            type=["csv"],
            help="Upload a CSV with at least two numeric columns"
        )
        if upload_file is not None:
            @st.cache_data
            def load_uploaded_data(file):
                return pd.read_csv(file)
            df = load_uploaded_data(upload_file)
            st.success("✅ File uploaded!")
        else:
            st.warning("⚠️ Please upload a CSV or use example dataset.")
            st.stop()
    
    st.markdown("---")
    
    # Dataset info in sidebar
    st.markdown("### 📊 Dataset Info")
    st.metric("Total Rows", df.shape[0])
    st.metric("Total Columns", df.shape[1])
    st.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))
    
    st.markdown("---")
    st.caption("⚡ Built with Streamlit • 3D Edition")

# ========================
# MAIN CONTENT
# ========================

# --- Dataset Preview ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📄 Dataset Preview")
st.dataframe(df, use_container_width=True, height=300)

if st.checkbox("📊 Show Statistics"):
    st.dataframe(df.describe().style.background_gradient(cmap="viridis"), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Model Configuration ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("⚙️ Model Configuration")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("❌ Need at least two numeric columns for regression")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    target = st.selectbox(
        "🎯 Target Variable (Dependent)",
        numeric_cols,
        help="Variable you want to predict"
    )

with col2:
    features = st.multiselect(
        "📌 Feature Columns (Independent)",
        [col for col in numeric_cols if col != target],
        default=[col for col in numeric_cols if col != target][:min(3, len(numeric_cols)-1)],
        help="Variables used to predict the target"
    )

if len(features) == 0:
    st.warning("⚠️ Please select at least one feature")
    st.stop()

# Test size slider
test_size = st.slider(
    "Test Set Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05
)
st.markdown('</div>', unsafe_allow_html=True)

# --- Data Preprocessing ---
df_model = df[features + [target]].dropna()
X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train Model ---
with st.spinner("🧠 Training model..."):
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

# --- Metrics with 3D cards ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📈 Model Performance")

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
    adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(features) - 1)
    st.markdown(f"""
    <div class="metric-3d">
        <div class="metric-value">{adj_r2:.3f}</div>
        <div class="metric-label">📐 Adj. R²</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- TABS for Visualizations ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["📊 Feature Importance", "🎯 Actual vs Predicted", "📉 Residual Analysis", "🌐 3D Visualization"])

# TAB 1: Feature Importance
with tab1:
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    })
    coef_df['Abs'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=coef_df['Coefficient'],
        y=coef_df['Feature'],
        orientation='h',
        marker=dict(
            color=coef_df['Coefficient'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Coefficient"),
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=coef_df['Coefficient'].round(3),
        textposition='outside',
        textfont=dict(color='rgba(255,255,255,0.7)')
    ))
    fig.update_layout(
        template='plotly_dark',
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.7)'),
        xaxis=dict(title='Coefficient Value', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        title='Feature Coefficients'
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: Actual vs Predicted
with tab2:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(
            size=12,
            color=y_pred - y_test,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title="Residual"),
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f"Actual: {a:.2f}<br>Predicted: {p:.2f}" for a, p in zip(y_test, y_pred)],
        hoverinfo='text',
        name='Predictions'
    ))
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='#4facfe', width=2, dash='dash'),
        name='Perfect Prediction',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.7)'),
        xaxis=dict(title='Actual Values', gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title='Predicted Values', gridcolor='rgba(255,255,255,0.05)'),
        title=f'Actual vs Predicted (R² = {r2:.3f})'
    )
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: Residual Analysis
with tab3:
    residuals = y_test - y_pred
    
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=('Residuals vs Predicted', 'Distribution of Residuals'))
    
    # Residuals vs Predicted
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            size=10,
            color=residuals,
            colorscale='RdBu',
            showscale=False,
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        text=[f"Predicted: {p:.2f}<br>Residual: {r:.2f}" for p, r in zip(y_pred, residuals)],
        hoverinfo='text',
        name='Residuals'
    ), row=1, col=1)
    
    fig.add_hline(y=0, line_dash="dash", line_color="#f5576c", row=1, col=1)
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=20,
        marker=dict(
            color='rgba(79, 172, 254, 0.6)',
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        name='Distribution'
    ), row=1, col=2)
    
    fig.add_vline(x=0, line_dash="dash", line_color="#f5576c", row=1, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(255,255,255,0.7)'),
        showlegend=False
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', row=1, col=1)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.05)', row=1, col=2)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: 3D Visualization (if 2+ features)
with tab4:
    if len(features) >= 2:
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Actual data points
        fig.add_trace(go.Scatter3d(
            x=X[features[0]],
            y=X[features[1]] if len(features) >= 2 else [0]*len(X),
            z=y,
            mode='markers',
            marker=dict(
                size=6,
                color=y,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=target),
                opacity=0.8
            ),
            name='Data Points',
            text=[f"{features[0]}: {a:.2f}<br>{features[1]}: {b:.2f}<br>{target}: {c:.2f}" 
                  for a, b, c in zip(X[features[0]], X[features[1]], y)],
            hoverinfo='text'
        ))
        
        # Create prediction surface (mesh grid)
        if len(features) >= 2:
            x_min, x_max = X[features[0]].min(), X[features[0]].max()
            y_min, y_max = X[features[1]].min(), X[features[1]].max()
            
            x_grid = np.linspace(x_min, x_max, 20)
            y_grid = np.linspace(y_min, y_max, 20)
            xx, yy = np.meshgrid(x_grid, y_grid)
            
            # Predict on grid
            grid_df = pd.DataFrame({
                features[0]: xx.ravel(),
                features[1]: yy.ravel()
            })
            # Add other features with mean values
            for f in features[2:]:
                grid_df[f] = X[f].mean()
            
            grid_scaled = scaler.transform(grid_df[features])
            z_grid = model.predict(grid_scaled).reshape(xx.shape)
            
            fig.add_trace(go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale='Viridis',
                opacity=0.6,
                showscale=False,
                name='Prediction Surface'
            ))
        
        fig.update_layout(
            template='plotly_dark',
            height=550,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='rgba(255,255,255,0.7)'),
            scene=dict(
                xaxis_title=features[0] if len(features) >= 1 else '',
                yaxis_title=features[1] if len(features) >= 2 else '',
                zaxis_title=target,
                bgcolor='rgba(0,0,0,0)',
                gridcolor='rgba(255,255,255,0.05)',
                showbackground=True,
                backgroundcolor='rgba(0,0,0,0.2)'
            ),
            title='3D Visualization: Features vs Target'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("💡 Add at least 2 features to see 3D visualization.")
st.markdown('</div>', unsafe_allow_html=True)

# --- Model Equation ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("📝 Model Equation")

intercept = model.intercept_
coefficients = model.coef_

equation = f"**{target} = {intercept:.3f}"
for i, (feature, coef) in enumerate(zip(features, coefficients)):
    if coef >= 0:
        equation += f" + {coef:.3f}·{feature}"
    else:
        equation += f" - {abs(coef):.3f}·{feature}"
equation += "**"

st.markdown(f"""
<div style="background: rgba(255,255,255,0.03); border-radius: 16px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.06);">
    <p style="font-size: 1.2rem; color: rgba(255,255,255,0.9); letter-spacing: 0.5px;">
        {equation}
    </p>
    <p style="font-size: 0.8rem; color: rgba(255,255,255,0.3); margin-top: 8px;">
        Intercept: {intercept:.3f} • Features: {len(features)}
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction Section ---
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
            key=f"pred_{feature}"
        )
        input_values.append(val)

if st.button("🔮 Predict", use_container_width=True):
    input_df = pd.DataFrame([input_values], columns=features)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(79, 172, 254, 0.15), rgba(240, 147, 251, 0.15)); 
                border-radius: 16px; padding: 24px; text-align: center; 
                border: 1px solid rgba(255,255,255,0.08); margin-top: 16px;">
        <p style="font-size: 0.8rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 2px;">
            Predicted {target}
        </p>
        <p style="font-size: 3rem; font-weight: 700; background: linear-gradient(135deg, #f093fb, #4facfe); 
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
            {prediction:.3f}
        </p>
        <p style="font-size: 0.9rem; color: rgba(255,255,255,0.3); margin-top: 8px;">
            ± {rmse:.3f} (RMSE)
        </p>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div style="text-align: center; padding: 30px 0 10px 0; border-top: 1px solid rgba(255,255,255,0.03);">
    <p style="color: rgba(255,255,255,0.15); font-size: 0.8rem; letter-spacing: 2px;">
        Built with ❤️ using Streamlit • 3D Linear Regression Pro
    </p>
</div>
""", unsafe_allow_html=True)
