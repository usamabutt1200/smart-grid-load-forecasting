import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import plotly.graph_objects as go

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Smart Grid Load Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS UI Theme
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 16px;
        color: #4B5563;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED MODEL LOADING FUNCTION
# ==========================================
@st.cache_resource
def load_forecasting_model():
    model_path = 'xgboost_energy_forecaster.pkl'
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model
    except FileNotFoundError:
        st.error(f"❌ Error: '{model_path}' not found! Please place your trained model file in the same directory.")
        return None

model = load_forecasting_model()

# ==========================================
# 3. SIDEBAR & USER CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://icons8.com", width=80)
    st.markdown("### **Control Panel**")
    st.write("Select the baseline parameters to forecast target electricity grid consumption constraints.")
    
    # Input widgets
    target_date = st.date_input("Target Forecast Date", datetime.date(2026, 10, 23))
    target_time = st.time_input("Target Forecast Time", datetime.time(12, 0))
    
    st.markdown("---")
    st.markdown("**Model Specifications:**")
    st.markdown("- Framework: `XGBoost Regressor` \n- Language Target: `English (Standard)`")

# ==========================================
# 4. APP MAIN BODY HEADER
# ==========================================
st.markdown('<div class="main-title">⚡ Smart Grid Energy Load Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production-grade predictive analytics pipeline using structural tree boosting to determine future grid demands.</div>', unsafe_allow_html=True)

if model is not None:
    # Combine selected date and time
    selected_datetime = datetime.datetime.combine(target_date, target_time)
    
    # Extract structural English temporal features for inference
    hour = selected_datetime.hour
    day = selected_datetime.day
    month = selected_datetime.month
    day_of_week = selected_datetime.weekday() # Monday=0, Sunday=6
    
    # Create single inference sample DataFrame
    input_data = pd.DataFrame([{
        'Hour': hour,
        'Day': day,
        'Month': month,
        'DayOfWeek': day_of_week
    }])
    
    # ==========================================
    # 5. CORE PREDICTION LOGIC
    # ==========================================
    prediction = model.predict(input_data)[0]
    
    # Simulate a full 24-hour cycle layout for visual graph enhancement
    hours_24 = list(range(24))
    daily_simulation_df = pd.DataFrame({
        'Hour': hours_24,
        'Day': [day] * 24,
        'Month': [month] * 24,
        'DayOfWeek': [day_of_week] * 24
    })
    hourly_predictions = model.predict(daily_simulation_df)

    # Display real-time Key Metrics Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <p style="margin:0; color:#6B7280; font-size:14px; font-weight:600;">FORECASTED POWER DEMAND</p>
                <h2 style="margin:5px 0 0 0; color:#1E3A8A;">{prediction:.2f} MW</h2>
                <p style="margin:0; color:#2563EB; font-size:12px;">At {selected_datetime.strftime('%H:%M %p')}</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        max_load = np.max(hourly_predictions)
        max_hour = np.argmax(hourly_predictions)
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #EF4444;">
                <p style="margin:0; color:#6B7280; font-size:14px; font-weight:600;">PEAK DEMAND TODAY</p>
                <h2 style="margin:5px 0 0 0; color:#DC2626;">{max_load:.2f} MW</h2>
                <p style="margin:0; color:#EF4444; font-size:12px;">Expected hour window: {max_hour}:00</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        min_load = np.min(hourly_predictions)
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #10B981;">
                <p style="margin:0; color:#6B7280; font-size:14px; font-weight:600;">BASE LOAD DEMAND</p>
                <h2 style="margin:5px 0 0 0; color:#059669;">{min_load:.2f} MW</h2>
                <p style="margin:0; color:#10B981; font-size:12px;">Minimum continuous infrastructure target</p>
            </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # 6. PREMIUM VISUALIZATIONS (PLOTLY CHART)
    # ==========================================
    st.markdown("### 📊 Predicted 24-Hour Energy Profile Curve")
    st.write(f"The structural analysis below showcases the full predicted energy consumption graph for **{target_date.strftime('%A, %B %d, %Y')}**:")

    fig = go.Figure()
    
    # 24-Hour line plot curve
    fig.add_trace(go.Scatter(
        x=hours_24, 
        y=hourly_predictions,
        mode='lines+markers',
        name='Predicted Curve',
        line=dict(color='#2563EB', width=3),
        marker=dict(size=6, color='#1E3A8A'),
        hovertemplate='Hour %{x}:00<br>Demand: %{y:.2f} MW<extra></extra>'
    ))
    
    # Highlight the specific single hour the user looked up
    fig.add_trace(go.Scatter(
        x=[hour],
        y=[prediction],
        mode='markers',
        name='Selected Timestamp Lookup',
        marker=dict(color='#EF4444', size=14, symbol='star'),
        hovertemplate='Selected Target!<br>Demand: %{y:.2f} MW<extra></extra>'
    ))

    fig.update_layout(
        xaxis=dict(title='Hour of Day (24H Format)', tickmode='linear', tick0=0, dtick=1),
        yaxis=dict(title='Energy Scale (Megawatts - MW)'),
        margin=dict(l=40, r=40, t=10, b=40),
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠️ Application components are locked until the primary machine learning binary file is loaded into the project core framework directory.")
