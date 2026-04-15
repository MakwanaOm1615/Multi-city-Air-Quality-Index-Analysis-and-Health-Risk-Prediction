# ==========================================
# IMPORTS
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
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Multi-city Air Quality Index Analysis and Health Risk Prediction",
    layout="wide", page_icon="🌍", initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important;}
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff, #f8fafc); padding: 15px;
        border-radius: 12px; border: 1px solid #e2e8f0; border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .alert-banner { background-color: #fef2f2; color: #991b1b; padding: 16px;
        border-left: 6px solid #dc2626; border-radius: 8px; margin-bottom: 24px; font-weight: 600;}
    .health-good { background-color: #f0fdf4; border-left: 6px solid #22c55e; padding: 16px; border-radius: 8px; }
    .health-mod  { background-color: #fefce8; border-left: 6px solid #eab308; padding: 16px; border-radius: 8px; }
    .health-poor { background-color: #fef2f2; border-left: 6px solid #ef4444; padding: 16px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Multi-city Air Quality Index Analysis and Health Risk Prediction")
st.markdown("Check air quality in your city, see predictions, and view health recommendations easily.")

# ==========================================
# NAVIGATION
# ==========================================
menu = option_menu(
    menu_title=None,
    options=["Overview", "Data Insights", "Compare Cities", "Predict AQI", "Forecast", "Health Advice"],
    icons=["house", "bar-chart", "map", "magic", "graph-up-arrow", "heart"],
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#ffffff", "border": "1px solid #e2e8f0", "border-radius": "10px"},
        "icon": {"color": "#64748b", "font-size": "14px"},
        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "color": "#1e293b", "font-weight": "600"},
        "nav-link-selected": {"background-color": "#3b82f6", "color": "white"},
    }
)
st.markdown("---")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def health_risk(aqi):
    if aqi <= 50:    return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 200: return "Poor"
    elif aqi <= 300: return "Very Poor"
    else:            return "Hazardous"

def get_aqi_color(v):
    if v <= 100: return "Green"
    elif v <= 300: return "Orange"
    else: return "Red"

CITY_COORDS = {
    "Delhi": (28.6139, 77.2090),         "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),     "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),       "Ahmedabad": (23.0225, 72.5714),
    "Kolkata": (22.5726, 88.3639),       "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),          "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),       "Patna": (25.5941, 85.1376),
    "Bhopal": (23.2599, 77.4126),        "Chandigarh": (30.7333, 76.7794),
    "Amritsar": (31.6340, 74.8723),      "Gurugram": (28.4595, 77.0266),
    "Guwahati": (26.1445, 91.7362),      "Coimbatore": (11.0168, 76.9558),
    "Kochi": (9.9312, 76.2673),          "Ernakulam": (9.9816, 76.2999),
    "Visakhapatnam": (17.6868, 83.2185), "Thiruvananthapuram": (8.5241, 76.9366),
    "Shillong": (25.5788, 91.8933),      "Aizawl": (23.7271, 92.7176),
    "Amaravati": (16.5730, 80.3582),     "Brajrajnagar": (21.8269, 83.9197),
    "Jorapokhar": (23.7244, 86.4144),    "Talcher": (20.9502, 85.2296),
    "Agra": (27.1767, 78.0081),          "Varanasi": (25.3176, 82.9739),
    "Surat": (21.1702, 72.8311),         "Nagpur": (21.1458, 79.0882),
    "Noida": (28.5355, 77.3910),         "Kanpur": (26.4499, 80.3319),
}

# Rename any column names to our standard names
def auto_map_columns(df):
    aliases = {
        "Date":  ["date", "day", "datetime", "time", "timestamp"],
        "City":  ["city", "stationid", "station_id", "station", "location", "area", "region", "place", "city_name"],
        "AQI":   ["aqi", "air_quality_index", "aqi_value"],
        "PM2.5": ["pm2.5", "pm25", "pm2_5"],
        "PM10":  ["pm10", "pm_10"],
        "NO2":   ["no2", "nitrogen_dioxide"],
        "SO2    ":   ["so2", "sulphur_dioxide", "sulfur_dioxide"],
        "CO":    ["co", "carbon_monoxide"],
    }   
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for std, opts in aliases.items():
        for o in opts:
            if o in lower_map and lower_map[o] != std:
                rename[lower_map[o]] = std
                break
    return df.rename(columns=rename)

# Clean and prepare any dataset — works for any CSV
def prepare_dataset(df):
    df = auto_map_columns(df)

    # Check required columns
    missing = [c for c in ["Date", "City", "AQI"] if c not in df.columns]
    if missing:
        return None, f"❌ Columns not found: {missing}. CSV must have Date, City, AQI."

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
    df = df.dropna(subset=['AQI'])

    # Smart duplicate fix:
    # Sirf tab groupby karo jab same City + same Date ke duplicate rows hon
    # Agar duplicates nahi hain to koi record drop nahi hoga
    if df.duplicated(subset=['City', 'Date']).any():
        existing_cols = [c for c in ['AQI','PM2.5','PM10','NO2','SO2','CO'] if c in df.columns]
        df = df.groupby(['City', 'Date'], as_index=False)[existing_cols].mean()

    df = df.sort_values(['City', 'Date'])
    df = df.ffill().bfill()

    # Add missing pollutant columns with 0
    for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Add coordinates from city name
    df['Lat'] = df['City'].apply(lambda x: CITY_COORDS.get(str(x).strip().title(), (None, None))[0])
    df['Lon'] = df['City'].apply(lambda x: CITY_COORDS.get(str(x).strip().title(), (None, None))[1])

    df["Health_Risk"] = df["AQI"].apply(health_risk)
    return df, None

# ==========================================
# LOAD DEFAULT DATA (only once at startup)
# ==========================================
@st.cache_data
def load_default_data():
    if os.path.exists("final_aqi_dataset.csv"):
        return pd.read_csv("final_aqi_dataset.csv")
    # Generate sample data if no file
    np.random.seed(42)
    cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Ahmedabad", "Pune", "Hyderabad"]
    records = []
    for city in cities:
        for date in pd.date_range("2023-01-01", "2023-12-31", freq="D"):
            pm25 = np.random.uniform(10, 250)
            pm10 = np.random.uniform(20, 300)
            no2  = np.random.uniform(5, 100)
            so2  = np.random.uniform(1, 50)
            co   = np.random.uniform(0.1, 5.0)
            aqi  = round((pm25 * 1.5) + (pm10 * 0.8) + (no2 * 0.5))
            records.append({"Date": date, "City": city, "PM2.5": round(pm25,2),
                "PM10": round(pm10,2), "NO2": round(no2,2), "SO2": round(so2,2),
                "CO": round(co,2), "AQI": aqi})
    return pd.DataFrame(records)

# Load default on first run
if 'df' not in st.session_state:
    raw, err = prepare_dataset(load_default_data())
    if raw is not None:
        st.session_state.df = raw
    else:
        st.error(err)
        st.stop()

# ==========================================
# ✅ UPLOAD SECTION
# KEY FIX: process uploaded file BEFORE st.rerun()
# and save to session_state FIRST, then rerun.
# ==========================================
with st.expander("📂 Upload Your Own Dataset", expanded=False):

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="uploader")

    if uploaded_file is not None:
        # Only process if it is a new file (compare by name)
        if st.session_state.get('last_uploaded') != uploaded_file.name:
            raw_uploaded = pd.read_csv(uploaded_file)
            cleaned, err = prepare_dataset(raw_uploaded)
            if err:
                st.error(err)
            else:
                st.session_state.df = cleaned
                st.session_state['last_uploaded'] = uploaded_file.name
                st.rerun()  # safe now — file name is saved so no infinite loop

    # Always show current dataset info
    st.success(f"Active dataset: {st.session_state.df['City'].nunique()} cities, {len(st.session_state.df)} records.")

    st.markdown("### Download Current Data")
    csv_out = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Processed CSV", data=csv_out, file_name='AirQualityData.csv', mime='text/csv')

st.markdown("---")

# Always read latest df from session
df = st.session_state.df

# ==========================================
# TRAIN ML MODEL ON CURRENT df
# ==========================================
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2']
df_model = df[features + ['AQI']].dropna()
if len(df_model) < 10:
    df_model = df[features + ['AQI']].fillna(0)

X = df_model[features]
y = df_model['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

# ==========================================
# PAGES
# ==========================================

# --- OVERVIEW ---
if menu == "Overview":
    st.header("📊 Overview")
    max_aqi = df["AQI"].max()
    if max_aqi >= 300:
        worst = df.loc[df["AQI"].idxmax(), "City"]
        st.markdown(f'<div class="alert-banner">⚠️ Very poor air quality in {worst} (AQI: {int(max_aqi)}). Take precautions.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Model RMSE", f"{rmse:.2f}")
    c2.metric("Cities Tracked", df["City"].nunique())
    c3.metric("Total Records", len(df))

    st.markdown("### 🗺️ Air Quality Map")
    map_df = df.groupby(["City","Lat","Lon"])["AQI"].mean().reset_index().dropna()
    if not map_df.empty:
        fig = px.scatter_mapbox(map_df, lat="Lat", lon="Lon", color="AQI", size="AQI",
            hover_name="City", color_continuous_scale="RdBu_r", size_max=30, zoom=3, mapbox_style="carto-positron")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Map not available — city coordinates not found.")

# --- DATA INSIGHTS ---
elif menu == "Data Insights":
    st.header("📈 Data Insights")
    city = st.selectbox("Select City", sorted(df["City"].unique()))
    city_df = df[df["City"] == city]
    t1, t2, t3 = st.tabs(["Timeline", "AQI Variations", "Health Risk"])
    with t1:
        st.plotly_chart(px.line(city_df, x="Date", y="AQI", title=f"AQI Over Time — {city}", markers=True), use_container_width=True)
    with t2:
        st.plotly_chart(px.box(df, x="City", y="AQI", color="City", title="AQI Box Plot All Cities"), use_container_width=True)
    with t3:
        risk_df = df.groupby(["City","Health_Risk"]).size().reset_index(name="Count")
        st.plotly_chart(px.bar(risk_df, x="City", y="Count", color="Health_Risk", barmode="group", title="Health Risk Distribution"), use_container_width=True)

# --- COMPARE CITIES ---
elif menu == "Compare Cities":
    st.header("⚖️ Compare Two Cities")
    cities_list = sorted(df["City"].unique())
    cA, cB = st.columns(2)
    with cA: city_a = st.selectbox("First City",  cities_list, index=0)
    with cB: city_b = st.selectbox("Second City", cities_list, index=min(1, len(cities_list)-1))
    df_a = df[df["City"] == city_a]
    df_b = df[df["City"] == city_b]
    r1, r2 = st.columns(2)
    r1.metric(f"{city_a} Avg AQI", f"{df_a['AQI'].mean():.1f}")
    r1.metric("Max AQI", f"{df_a['AQI'].max():.1f}")
    r2.metric(f"{city_b} Avg AQI", f"{df_b['AQI'].mean():.1f}")
    r2.metric("Max AQI", f"{df_b['AQI'].max():.1f}")
    st.plotly_chart(px.line(pd.concat([df_a, df_b]), x="Date", y="AQI", color="City", title="AQI Comparison"), use_container_width=True)

# --- PREDICT AQI ---
elif menu == "Predict AQI":
    st.header("🔮 Predict Air Quality")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Enter Pollutant Levels:")
        pm25 = st.number_input("PM2.5", min_value=0.0, value=50.0)
        pm10 = st.number_input("PM10",  min_value=0.0, value=75.0)
        no2  = st.number_input("NO2",   min_value=0.0, value=30.0)
        co   = st.number_input("CO",    min_value=0.0, value=1.0)
        so2  = st.number_input("SO2",   min_value=0.0, value=15.0)
    with c2:
        st.markdown("### Result:")
        if st.button("Predict AQI", type="primary"):
            val = model.predict(np.array([[pm25, pm10, no2, co, so2]]))[0]
            st.metric("Predicted AQI", f"{val:.2f}")
            st.info(f"Health Status: {health_risk(val)}")
            fig = go.Figure(go.Indicator(mode="gauge+number", value=val,
                gauge={'axis': {'range': [0,500]}, 'bar': {'color': get_aqi_color(val)},
                       'steps': [{'range':[0,100],'color':'#f8fafc'},{'range':[100,300],'color':'#e2e8f0'},{'range':[300,500],'color':'#cbd5e1'}]}))
            fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)

# --- FORECAST ---
elif menu == "Forecast":
    st.header("📅 AQI Forecast")
    city = st.selectbox("Select City", sorted(df["City"].unique()))
    days = st.slider("Days to Forecast:", 1, 30, 7)

    ts = df[df["City"]==city].sort_values("Date").set_index("Date")
    ts = ts[~ts.index.duplicated(keep='last')]
    try:
        ts = ts.asfreq('D')
    except:
        ts = ts.resample('D').mean()
    ts["AQI"] = ts["AQI"].ffill().bfill()
    ts.index = ts.index + (pd.Timestamp(datetime.datetime.today().date()) - ts.index[-1])

    try:
        forecast = ARIMA(ts["AQI"], order=(2,1,1)).fit().forecast(steps=days)
        if forecast.isna().any():
            raise ValueError("NaN in forecast")
    except:
        last = ts["AQI"].iloc[-1]
        forecast = pd.Series(
            [last + np.random.uniform(-5,5) for _ in range(days)],
            index=pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=days)
        )

    fig = px.line(ts, y="AQI", title=f"{city} — AQI Forecast for Next {days} Days")
    fig.add_scatter(x=forecast.index, y=forecast.values, mode="lines+markers", name="Forecast", line=dict(color="#3b82f6", width=3))
    st.plotly_chart(fig, use_container_width=True)
    fdf = forecast.reset_index()
    fdf.columns = ["Date","Predicted AQI"]
    fdf["Date"] = fdf["Date"].dt.strftime('%A, %d %B %Y')
    fdf["Predicted AQI"] = fdf["Predicted AQI"].round(2)
    st.dataframe(fdf, use_container_width=True)

# --- HEALTH ADVICE ---
elif menu == "Health Advice":
    st.header("❤️ Health Advice")
    st.markdown("""
    <div class="health-good"><strong>🟢 Good (AQI 0–100)</strong><br>Air is clean. Safe for all outdoor activities.</div><br>
    <div class="health-mod"><strong>🟡 Moderate / Poor (AQI 101–300)</strong><br>Sensitive groups should limit prolonged outdoor activity.</div><br>
    <div class="health-poor"><strong>🔴 Very Poor / Hazardous (AQI 301+)</strong><br>Health emergency! Stay indoors. Keep windows closed.</div>
    """, unsafe_allow_html=True)