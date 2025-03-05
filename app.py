import streamlit as st
import pickle
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px

# For Open-Meteo historical API
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --------------------------------------
# Setup cache and retry for historical API calls
# --------------------------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --------------------------------------
# Streamlit Page Configuration
# --------------------------------------
st.set_page_config(page_title="Smart Travel Planner", layout="wide")

# --------------------------------------
# 1. Load Pre-trained KNN Model (trained on full historical DataFrame)
# --------------------------------------
@st.cache_resource
def load_knn_model():
    with open("Knn.pkl", "rb") as file:
        model = joblib.load(file)
    return model

model = load_knn_model()

# --------------------------------------
# 2. Predefined Destinations with Coordinates
# --------------------------------------
destinations = {
    "Bali": ("-8.34", "115.09"),
    "Bangkok": ("13.73", "100.52"),
    "Barcelona": ("41.38", "2.15"),
    "Berlin": ("52.52", "13.41"),
    "Cape Town": ("-33.91", "18.42"),
    "Dubai": ("25.06", "55.17"),
    "Istanbul": ("41.16", "28.76"),
    "Lima": ("-12.04", "-77.04"),
    "London": ("51.51", "-0.11"),
    "Los Angeles": ("34.05", "-118.24"),
    "Melbourne": ("-37.81", "144.96"),
    "New York City": ("40.71", "-74.00"),
    "Paris": ("48.85", "2.35"),
    "Prague": ("50.08", "14.42"),
    "Queenstown": ("-45.03", "168.66"),
    "Rio De Janeiro": ("-22.90", "-43.17"),
    "Rome": ("41.90", "12.49"),
    "Seoul": ("37.56", "126.97"),
    "Singapore": ("1.36", "103.8"),
    "Sydney": ("-33.86", "151.20"),
    "Tokyo": ("35.68", "139.69"),
    "Vienna": ("48.20", "16.37")
}

# --------------------------------------
# 3. Weather API (RapidAPI) Configuration & Helper Function (Forecast)
# --------------------------------------
WEATHER_API_URL = "https://weatherapi-com.p.rapidapi.com/forecast.json"

@st.cache_data(show_spinner=False)
def get_weather_data(query, days=7):
    querystring = {"q": query, "days": days}
    headers = {
        "x-rapidapi-key": "b4775bf9d4msh4c956202bcc2d52p17aacejsn4e67ce5ec4f0",  # Replace with your key
        "x-rapidapi-host": "weatherapi-com.p.rapidapi.com"
    }
    response = requests.get(WEATHER_API_URL, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching forecast data: {response.status_code}")
        return None

# --------------------------------------
# 4. Historical Data Retrieval Function (using Open-Meteo)
# --------------------------------------
def get_full_historical_dataframe(lat, lon, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
            "precipitation_probability", "precipitation", "rain", "showers", "snowfall",
            "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover",
            "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility",
            "evapotranspiration", "et0_fao_evapotranspiration", "vapour_pressure_deficit",
            "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
            "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
            "wind_gusts_10m", "temperature_80m", "temperature_120m", "temperature_180m",
            "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm",
            "soil_temperature_54cm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm",
            "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"
        ]
    }
    responses = openmeteo.weather_api(url, params=params)
    if responses:
        response = responses[0]
        hourly = response.Hourly()
        date_range = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
        variable_names = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
            "precipitation_probability", "precipitation", "rain", "showers", "snowfall",
            "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover",
            "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility",
            "evapotranspiration", "et0_fao_evapotranspiration", "vapour_pressure_deficit",
            "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
            "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
            "wind_gusts_10m", "temperature_80m", "temperature_120m", "temperature_180m",
            "soil_temperature_0cm", "soil_temperature_6cm", "soil_temperature_18cm",
            "soil_temperature_54cm", "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm",
            "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "soil_moisture_27_to_81cm"
        ]
        data_dict = {"date": date_range}
        for i, var in enumerate(variable_names):
            data_dict[var] = hourly.Variables(i).ValuesAsNumpy()
        df_full = pd.DataFrame(data=data_dict)
        return df_full
    else:
        st.error("Error fetching full historical data.")
        return None

# --------------------------------------
# 5. Helper: Optimal Travel Date Suggestion (Forecast-based)
# --------------------------------------
def find_optimal_travel_date(forecast, user_date):
    best_score = -np.inf
    best_date = None
    for day in forecast:
        date_str = day["date"]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        if date_obj >= user_date:
            avg_temp = day["day"]["avgtemp_c"]
            precip = day["day"].get("totalprecip_mm", 0)
            score = avg_temp - 2 * precip  # heuristic: higher temp, lower precip better
            if score > best_score:
                best_score = score
                best_date = date_str
    return best_date

# --------------------------------------
# 6. Streamlit App Layout & User Inputs
# --------------------------------------
st.title("🌍 Smart Travel Planner")
st.subheader("Your AI-powered companion for personalized travel recommendations")

st.sidebar.header("Plan Your Trip")
selected_destination = st.sidebar.selectbox("Select your destination", list(destinations.keys()))
lat, lon = destinations[selected_destination]
location_query = f"{lat},{lon}"

user_travel_date = st.sidebar.date_input("Select your travel date", datetime.today() + timedelta(days=1))
preferred_climate = st.sidebar.selectbox("Preferred Weather Condition", ["Sunny", "Rainy", "Mild", "Cold"])

show_historical = st.sidebar.checkbox("Show Full Historical Weather Data")
if show_historical:
    st.sidebar.markdown("### Historical Data Settings")
    hist_start = st.sidebar.date_input("Historical Start Date", datetime(2024, 12, 1))
    hist_end = st.sidebar.date_input("Historical End Date", datetime(2025, 3, 4))

# --------------------------------------
# 7. Main Button: Get Travel Recommendations
# --------------------------------------
if st.sidebar.button("Get Travel Recommendations"):
    # --- Forecast Data Retrieval via WeatherAPI ---
    weather_data = get_weather_data(location_query, days=7)
    if weather_data:
        current_weather = weather_data.get("current", {})
        location_info = weather_data.get("location", {})
        forecast = weather_data.get("forecast", {}).get("forecastday", [])
        
        st.subheader(f"Weather Insights for {selected_destination}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Temperature", f"{current_weather.get('temp_c', 'N/A')} °C")
        col2.metric("Humidity", f"{current_weather.get('humidity', 'N/A')} %")
        col3.metric("Wind Speed", f"{current_weather.get('wind_kph', 'N/A')} kph")
        col4.metric("Local Time", location_info.get("localtime", "N/A"))
        
        alerts = weather_data.get("alerts", {}).get("alert", [])
        if alerts:
            st.error("Severe Weather Alerts:")
            for alert in alerts:
                st.warning(f"{alert.get('headline', '')}: {alert.get('desc', '')}")
        else:
            st.info("No severe weather alerts.")
        
        optimal_date = find_optimal_travel_date(forecast, user_travel_date)
        st.markdown(f"### Optimal Travel Date Suggestion: **{optimal_date}**")
        
        forecast_df = pd.DataFrame([{
            "Date": f["date"],
            "Max Temp": f["day"]["maxtemp_c"],
            "Min Temp": f["day"]["mintemp_c"],
            "Avg Temp": f["day"]["avgtemp_c"],
            "Condition": f["day"]["condition"]["text"]
        } for f in forecast])
        fig = px.line(forecast_df, x="Date", y=["Max Temp", "Min Temp", "Avg Temp"],
                      title="7-Day Temperature Forecast", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # --- Historical Data Retrieval ---
        df_hist = None
        if show_historical:
            df_hist = get_full_historical_dataframe(lat, lon, hist_start.strftime("%Y-%m-%d"), hist_end.strftime("%Y-%m-%d"))
            if df_hist is not None:
                st.subheader("Full Historical Weather Data")
                st.dataframe(df_hist)
                st.line_chart(df_hist.set_index("date")["temperature_2m"])
            else:
                st.error("Failed to retrieve historical data.")
        
        # --- AI-Based Travel Comfort Prediction using full historical features ---
        if df_hist is not None:
            st.subheader("AI-Based Travel Comfort Prediction")
            timestamp_options = df_hist["date"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
            selected_timestamp = st.selectbox("Select a timestamp for prediction", timestamp_options)
            selected_row = df_hist[df_hist["date"].dt.strftime("%Y-%m-%d %H:%M:%S") == selected_timestamp]
            if not selected_row.empty:
                model_input = selected_row.drop(columns=["date"]).to_numpy()
                st.write("Model input shape:", model_input.shape)
                st.write("Model input preview:", model_input)
                if model_input.shape[1] != model.n_features_in_:
                    st.error(f"Model expects {model.n_features_in_} features, but got {model_input.shape[1]}. Check your feature engineering.")
                else:
                    prediction = model.predict(model_input)
                    st.write("Raw prediction output:", prediction)
                    comfort_map = {0: "Low", 1: "Medium", 2: "High"}
                    predicted_comfort = comfort_map.get(prediction[0], "Unknown")
                    st.markdown(f"**Predicted Travel Comfort for {selected_timestamp}: {predicted_comfort}**")
            else:
                st.error("No data available for the selected timestamp.")
    else:
        st.error("Could not retrieve forecast data. Please try again.")
