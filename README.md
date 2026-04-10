# 🌍 Multi-city AQI Analysis & Health Risk Prediction

An interactive web application built with **Streamlit** to monitor, analyze, and predict the Air Quality Index (AQI) for various cities. This project aims to provide actionable health insights based on environmental pollutant data.

## 🚀 Key Features
* **Overview:** Real-time metrics for major pollutants (PM2.5, PM10, NO2, etc.).
* **Data Insights:** Deep dive into historical trends and pollutant distributions.
* **Compare Cities:** Side-by-side analysis of air quality across different urban centers.
* **Predict AQI:** Machine Learning model (Random Forest) to predict AQI based on current pollutant levels.
* **Forecast:** Time-series forecasting using ARIMA to predict future air quality trends.
* **Health Advice:** Personalized safety recommendations based on current AQI categories.

## 🛠️ Built With
* **Language:** Python
* **Web Framework:** [Streamlit](https://streamlit.io/)
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Plotly, Seaborn
* **Machine Learning:** Scikit-learn (Random Forest), Statsmodels (ARIMA)

## 📁 Project Structure

```text
OEP/
├── .streamlit/
│   └── config.toml          # Configuration for Light Mode theme
├── app.py                   # Main application script
├── final_aqi_dataset.csv    # Historical pollutant dataset
├── requirements.txt         # Python dependencies for deployment
└── README.md                # Project documentation
```
## ⚙️ Local Setup
1. Clone the repository: `git clone https://github.com`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---
*Disclaimer: This tool provides general informational data. For health concerns related to air quality, always consult a medical professional.*
