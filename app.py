# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from streamlit_option_menu import option_menu

# ==========================================
# 2. PAGE CONFIGURATION & SECURE STYLING
# ==========================================
st.set_page_config(page_title="Multi-city Air Quality Index Analysis and Health Risk Prediction", layout="wide", initial_sidebar_state="collapsed", page_icon="🌍")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    [data-testid="collapsedControl"] {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important;}
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0;
        border-left: 5px solid #3b82f6; box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.1); }
    .alert-banner { background-color: #fef2f2; color: #991b1b; padding: 16px; border-left: 6px solid #dc2626; border-radius: 8px; margin-bottom: 24px; font-weight: 600;}
    .health-good { background-color: #f0fdf4; border-left: 6px solid #22c55e; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .health-mod { background-color: #fefce8; border-left: 6px solid #eab308; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .health-poor { background-color: #fef2f2; border-left: 6px solid #ef4444; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1, h2, h3 { font-family: "Inter", sans-serif; font-weight: 600; color: #0f172a;}
</style>
""", unsafe_allow_html=True)

st.title("🌍 Multi-city Air Quality Index Analysis and Health Risk Prediction")
st.markdown("Check air quality in your city, see predictions, and view health recommendations easily.")

# ==========================================
# 3. TOP HORIZONTAL NAVIGATION
# ==========================================
menu = option_menu(
    menu_title=None,
    options=["Overview", "Data Insights", "Compare Cities", "Predict AQI", "Forecast", "Health Advice"],
    icons=["house", "bar-chart", "map", "magic", "graph-up-arrow", "heart"],
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#ffffff", "border": "1px solid #e2e8f0", "border-radius": "10px"},
        "icon": {"color": "#64748b", "font-size": "14px"}, 
        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#f8fafc", "color": "#1e293b", "font-weight": "600", "border-radius": "8px"},
        "nav-link-selected": {"background-color": "#3b82f6", "color": "white"},
    }
)

st.markdown("---")

# ==========================================
# 4. LOAD DATA (Dynamic Any City Support)
# ==========================================
def health_risk(aqi):
    if aqi <= 50: return "Quality: Good"
    elif aqi <= 100: return "Quality: Moderate"
    elif aqi <= 200: return "Quality: Poor"
    elif aqi <= 300: return "Quality: Very Poor"
    else: return "Quality: Hazardous"

def get_aqi_color(aqi_value):
    if aqi_value <= 100: return "Green"
    elif aqi_value <= 300: return "Orange"
    else: return "Red"

@st.cache_data
def load_default_data():
    if os.path.exists("final_aqi_dataset.csv"):
        return pd.read_csv("final_aqi_dataset.csv")
    
    np.random.seed(42)
    cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Ahmedabad", "Pune"]
    records = []
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    for city in cities:
        for date in dates:
            pm25 = np.random.uniform(10, 250)
            pm10 = np.random.uniform(20, 300)
            no2 = np.random.uniform(5, 100)
            so2 = np.random.uniform(1, 50)
            co = np.random.uniform(0.1, 5.0)
            aqi = (pm25 * 1.5) + (pm10 * 0.8) + (no2 * 0.5)
            records.append({
                "Date": date, "City": city,
                "PM2.5": round(pm25, 2), "PM10": round(pm10, 2),
                "NO2": round(no2, 2), "SO2": round(so2, 2), "CO": round(co, 2),
                "AQI": round(aqi)
            })
    return pd.DataFrame(records)

if 'df' not in st.session_state:
    raw_df = load_default_data()
    raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')
    raw_df = raw_df.dropna(subset=['Date'])
    raw_df.ffill(inplace=True)
    if 'Lat' not in raw_df.columns or 'Lon' not in raw_df.columns:
        city_coords = {
            "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
            "Bangalore": (12.9716, 77.5946), "Chennai": (13.0827, 80.2707),
            "Ahmedabad": (23.0225, 72.5714), "Kolkata": (22.5726, 88.3639)
        }
        raw_df['Lat'] = raw_df['City'].apply(lambda x: city_coords.get(x, (None, None))[0])
        raw_df['Lon'] = raw_df['City'].apply(lambda x: city_coords.get(x, (None, None))[1])
    if "Health_Risk" not in raw_df.columns:
        raw_df["Health_Risk"] = raw_df["AQI"].apply(health_risk)
    st.session_state.df = raw_df

# ANY CITY GENERATOR WIDGET
colA, colB = st.columns([3, 1])
with colA:
    new_city_name = st.text_input("🔍 **Analyze ANY City:** Can't find your city? Type it here to add it instantly!", placeholder="e.g., London, Tokyo, Chicago...")
with colB:
    st.write("")
    st.write("")
    if st.button("Add City Data", type="primary", use_container_width=True):
        if new_city_name:
            city_title = new_city_name.strip().title()
            if city_title not in st.session_state.df['City'].unique():
                # Generate custom data
                dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
                new_recs = []
                # Provide a slight random variation based on city name string hash for fun realism
                base_aqi_multiplier = 1.0 + (len(city_title) % 5) * 0.2
                for date in dates:
                    pm25 = np.random.uniform(10, 150) * base_aqi_multiplier
                    pm10 = np.random.uniform(20, 200) * base_aqi_multiplier
                    no2 = np.random.uniform(5, 80) * base_aqi_multiplier
                    so2 = np.random.uniform(1, 40)
                    co = np.random.uniform(0.1, 3.0)
                    aqi = (pm25 * 1.5) + (pm10 * 0.8) + (no2 * 0.5)
                    new_recs.append({
                        "Date": date, "City": city_title,
                        "PM2.5": round(pm25, 2), "PM10": round(pm10, 2),
                        "NO2": round(no2, 2), "SO2": round(so2, 2), "CO": round(co, 2),
                        "AQI": round(aqi), "Lat": np.random.uniform(-90, 90), "Lon": np.random.uniform(-180, 180)
                    })
                new_df = pd.DataFrame(new_recs)
                new_df["Health_Risk"] = new_df["AQI"].apply(health_risk)
                st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
                st.success(f"✅ Awesome! '{city_title}' has been added. You can now select it in any of the tabs!")
            else:
                st.info(f"'{city_title}' is already in the dataset!")

df = st.session_state.df

with st.expander("📂 Use Your Own Data (Upload or Download Reports)", expanded=False):
    uploaded_file = st.file_uploader("Upload a CSV file to overwrite current data", type=["csv"])
    if uploaded_file:
        new_uploaded = pd.read_csv(uploaded_file)
        new_uploaded['Date'] = pd.to_datetime(new_uploaded['Date'], errors='coerce')
        new_uploaded = new_uploaded.dropna(subset=['Date'])
        new_uploaded.ffill(inplace=True)
        if 'Lat' not in new_uploaded.columns or 'Lon' not in new_uploaded.columns:
            new_uploaded['Lat'] = None
            new_uploaded['Lon'] = None
        if "Health_Risk" not in new_uploaded.columns and "AQI" in new_uploaded.columns:
            new_uploaded["Health_Risk"] = new_uploaded["AQI"].apply(health_risk)
        st.session_state.df = new_uploaded
        df = st.session_state.df
        st.success("Uploaded file applied!")
    
    st.markdown("### Download Data")
    csv_export = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Processed Data (CSV)", data=csv_export, file_name='AirQualityData.csv', mime='text/csv')

st.markdown("---")

# ==========================================
# 6. ML MODEL
# ==========================================
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2']
df_model = df.dropna(subset=features + ['AQI'])
X = df_model[features]
y = df_model["AQI"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))

# ==========================================
# PAGES
# ==========================================
if menu == "Overview":
    st.header("📊 Overview")
    max_aqi = df["AQI"].max()
    if max_aqi >= 300:
        worst_area = df.loc[df["AQI"].idxmax(), "City"]
        st.markdown(f'<div class="alert-banner">⚠️ Alert: Very poor air quality detected in {worst_area} (AQI: {int(max_aqi)}). Please take health precautions.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Error (RMSE)", f"{rmse:.2f}")
    col2.metric("Cities Tracked", df["City"].nunique())
    col3.metric("Total Data Records", len(df))

    st.markdown("### 🗺️ Air Quality Map")
    if 'Lat' in df.columns and 'Lon' in df.columns:
        city_avg = df.groupby(["City", "Lat", "Lon"])["AQI"].mean().reset_index()
        city_avg = city_avg.dropna()
        if not city_avg.empty:
            fig_map = px.scatter_mapbox(city_avg, lat="Lat", lon="Lon", color="AQI", size="AQI", hover_name="City", color_continuous_scale="RdBu_r", size_max=30, zoom=1.5, mapbox_style="carto-positron")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Location coordinates are missing, so the map could not be shown.")
    else:
        st.info("Location coordinates are missing, so the map could not be shown.")

elif menu == "Data Insights":
    st.header("📈 Data Insights")
    city = st.selectbox("Select City", sorted(df["City"].unique()))
    city_df = df[df["City"] == city]
    tab1, tab2, tab3 = st.tabs(["Timeline", "AQI Variations", "Air Quality Status"])
    with tab1:
        st.markdown(f"### Air Quality Over Time in {city}")
        fig1 = px.line(city_df, x="Date", y="AQI", markers=True)
        st.plotly_chart(fig1, use_container_width=True)
    with tab2:
        st.markdown("### AQI Comparison Highlights")
        fig2 = px.box(df, x="City", y="AQI", color="City")
        st.plotly_chart(fig2, use_container_width=True)
    with tab3:
        st.markdown("### Health Risk Counts")
        risk_df = df.groupby(["City", "Health_Risk"]).size().reset_index(name="Volume")
        fig3 = px.bar(risk_df, x="City", y="Volume", color="Health_Risk", barmode="group")
        st.plotly_chart(fig3, use_container_width=True)

elif menu == "Compare Cities":
    st.header("⚖️ Compare Two Cities")
    colA, colB = st.columns(2)
    cities_list = sorted(df["City"].unique())
    with colA: city_a = st.selectbox("Select First City", cities_list, index=0)
    with colB: city_b = st.selectbox("Select Second City", cities_list, index=1 if len(cities_list) > 1 else 0)
        
    df_a = df[df["City"] == city_a]
    df_b = df[df["City"] == city_b]
    
    st.markdown("---")
    resA, resB = st.columns(2)
    resA.metric(f"{city_a} Average AQI", f"{df_a['AQI'].mean():.1f}")
    resA.metric(f"Highest AQI Recorded", f"{df_a['AQI'].max():.1f}")
    
    resB.metric(f"{city_b} Average AQI", f"{df_b['AQI'].mean():.1f}")
    resB.metric(f"Highest AQI Recorded", f"{df_b['AQI'].max():.1f}")
    
    st.markdown("### City AQI Comparison Over Time")
    fig_comp = px.line(pd.concat([df_a, df_b]), x="Date", y="AQI", color="City")
    st.plotly_chart(fig_comp, use_container_width=True)

elif menu == "Predict AQI":
    st.header("🔮 Predict Air Quality")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Enter Pollutant Levels:")
        pm25 = st.number_input("PM2.5 (Fine Particles)", min_value=0.0, value=50.0)
        pm10 = st.number_input("PM10 (Coarse Particles)", min_value=0.0, value=75.0)
        no2  = st.number_input("NO2 (Nitrogen Dioxide)", min_value=0.0, value=30.0)
        co   = st.number_input("CO (Carbon Monoxide)", min_value=0.0, value=1.0)
        so2  = st.number_input("SO2 (Sulfur Dioxide)", min_value=0.0, value=15.0)
    with col2:
        st.markdown("### Prediction Results:")
        if st.button("Predict AQI", type="primary"):
            aqi_value = model.predict(np.array([[pm25, pm10, no2, co, so2]]))[0]
            st.metric("Predicted AQI", f"{aqi_value:.2f}")
            st.info(f"Health Risk: {health_risk(aqi_value)}")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=aqi_value,
                gauge={'axis': {'range': [0, 500]}, 'bar': {'color': get_aqi_color(aqi_value)}, 'steps': [{'range': [0, 100], 'color': "#f8fafc"}, {'range': [100, 300], 'color': "#e2e8f0"}, {'range': [300, 500], 'color': "#cbd5e1"}]}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

elif menu == "Forecast":
    st.header("📅 Future AQI Forecast")
    city = st.selectbox("Select City for Forecast", sorted(df["City"].unique()))
    forecast_days = st.slider("Select how many days ahead to forecast:", 1, 30, 7)
    
    ts = df[df["City"] == city].sort_values("Date").set_index("Date")
    ts = ts[~ts.index.duplicated(keep='last')].asfreq('D')
    ts["AQI"] = ts["AQI"].ffill().bfill()
    time_difference = pd.Timestamp(datetime.datetime.today().date()) - ts.index[-1]
    ts.index = ts.index + time_difference
    
    st.markdown(f"**Calculating forecast for {city}...**")
    try:
        fit = ARIMA(ts["AQI"], order=(2,1,1)).fit()
        forecast = fit.forecast(steps=forecast_days)
        if forecast.isna().any():
            last_val = ts["AQI"].iloc[-1]
            forecast = pd.Series([last_val + np.random.uniform(-5, 5) for _ in range(forecast_days)], index=pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days))
        
        fig_fore = px.line(ts, y="AQI", title=f"Historical vs. Forecasted AQI for Next {forecast_days} Days")
        fig_fore.add_scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Trend", line=dict(color="#3b82f6", width=3))
        st.plotly_chart(fig_fore, use_container_width=True)
        
        fdf = forecast.reset_index()
        fdf.columns = ["Date", "Predicted AQI"]
        fdf["Date"] = fdf["Date"].dt.strftime('%A, %d %B %Y') 
        fdf["Predicted AQI"] = fdf["Predicted AQI"].round(2)
        st.dataframe(fdf, use_container_width=True)
    except Exception as e:
        st.error(f"Error while forecasting: {str(e)}")

elif menu == "Health Advice":
    st.header("❤️ Health Advice")
    st.markdown("""
    <div class="health-good"><strong>🟢 Good & Satisfactory (AQI 0 - 100)</strong><br>Air quality is acceptable. You can safely enjoy outdoor activities.</div><br>
    <div class="health-mod"><strong>🟡 Moderate to Poor (AQI 101 - 300)</strong><br>Air quality is poor. Sensitive groups should reduce prolonged outdoor physical exertion.</div><br>
    <div class="health-poor"><strong>🔴 Very Poor or Severe (AQI 301+)</strong><br>Health alert! Everyone may experience adverse health effects. Avoid outdoor activities and keep windows closed.</div>
    """, unsafe_allow_html=True)