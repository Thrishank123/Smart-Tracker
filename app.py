import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.set_page_config(page_title="Smart Travel Planner", layout="wide")


WEATHER_API_URL = "https://weatherapi-com.p.rapidapi.com/forecast.json"

@st.cache_data(show_spinner=False)
def get_weather_data(query, days=7):
    querystring = {"q": query, "days": days}
    headers = {
        "x-rapidapi-key": "b4775bf9d4msh4c956202bcc2d52p17aacejsn4e67ce5ec4f0",  # Replace with your actual key
        "x-rapidapi-host": "weatherapi-com.p.rapidapi.com"
    }
    response = requests.get(WEATHER_API_URL, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching forecast data: {response.status_code}")
        return None
    
def find_optimal_travel_date(forecast, user_date):
    """
    Suggests an optimal travel date based on a simple heuristic:
    selects the forecast day (on or after user_date) with the highest score,
    where score = avgtemp - 2 * total precipitation.
    """
    best_score = -np.inf
    best_date = None
    for day in forecast:
        date_str = day["date"]
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        if date_obj >= user_date:
            avg_temp = day["day"]["avgtemp_c"]
            precip = day["day"].get("totalprecip_mm", 0)
            score = avg_temp - 2 * precip
            if score > best_score:
                best_score = score
                best_date = date_str
    return best_date
# For Open-Meteo historical API
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --------------------------------------
# Custom CSS for better mobile UI (adjust as needed)
# --------------------------------------
st.markdown("""
    <style>
    @media only screen and (max-width: 600px) {
        .css-1d391kg { font-size: 16px; }
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------------------
# Setup cache and retry for historical API calls
# --------------------------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --------------------------------------
# Streamlit Page Configuration
# --------------------------------------

# --------------------------------------
# 1. Load Pre-trained KNN Model (trained on processed historical data)
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
# 3. Historical Data Retrieval Function using Open-Meteo
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
# 4b. Process the Forecast Data (for prediction)
# Here we create a simple processing function that extracts and scales key features.
# Adjust this processing to match the training pipeline of your model.
def process_forecast_data(forecast):
    # For each forecast day, extract a set of key features:
    # average temperature, total precipitation, max wind speed, average humidity.
    data = []
    for day in forecast:
        avgtemp = day["day"]["avgtemp_c"]
        totalprecip = day["day"].get("totalprecip_mm", 0)
        maxwind = day["day"].get("maxwind_kph", 0)
        avghumidity = day["day"].get("avghumidity", 0)
        data.append([avgtemp, totalprecip, maxwind, avghumidity])
    df = pd.DataFrame(data, columns=["avgtemp", "totalprecip", "maxwind", "avghumidity"])
    # Scale features as per training; here we use MinMax scaling as an example.
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

# --------------------------------------
# 5. Streamlit App Layout & User Inputs
# --------------------------------------
def Processed(df):
    # ---------------------------
    # Date Conversion and Time-Based Features
    # ---------------------------
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['day_of_week'] = df['date'].dt.dayofweek   # Monday=0, Sunday=6
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # ---------------------------
    # Define Season Based on Country and Month
    # ---------------------------
    def get_season(row):
        month = row['month']
        country = str(row['Country']).strip().lower() if 'Country' in row else ""
        southern_countries = ['australia', 'new zealand', 'south africa', 'argentina', 'chile', 'brazil']
        if country in southern_countries:
            if month in [12, 1, 2]:
                return 'Summer'
            elif month in [3, 4, 5]:
                return 'Autumn'
            elif month in [6, 7, 8]:
                return 'Winter'
            elif month in [9, 10, 11]:
                return 'Spring'
        else:
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            elif month in [9, 10, 11]:
                return 'Autumn'
    df['season'] = df.apply(get_season, axis=1)

    # ---------------------------
    # Handle Missing Data
    # ---------------------------
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # ---------------------------
    # Drop Less Relevant Columns
    # ---------------------------
    soil_columns = [col for col in df.columns if "soil" in col.lower()]
    if soil_columns:
        df.drop(columns=soil_columns, inplace=True)

    # ---------------------------
    # Advanced Feature Engineering
    # ---------------------------
    wind_cols = [col for col in ['wind_speed_10m', 'wind_speed_80m', 'wind_speed_120m', 'wind_speed_180m'] if col in df.columns]
    if wind_cols:
        df['avg_wind_speed'] = df[wind_cols].mean(axis=1)
    else:
        df['avg_wind_speed'] = np.nan

    cloud_cols = [col for col in ['cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high'] if col in df.columns]
    if cloud_cols:
        df['avg_cloud_cover'] = df[cloud_cols].mean(axis=1)

    if 'apparent_temperature' in df.columns and 'temperature_2m' in df.columns:
        df['temp_difference'] = df['apparent_temperature'] - df['temperature_2m']

    severity_features = [col for col in ['precipitation', 'snowfall', 'showers'] if col in df.columns]
    if severity_features:
        df['weather_severity_index'] = df[severity_features].sum(axis=1)
    else:
        df['weather_severity_index'] = 0

    tci_weights = {
        'temperature_2m': 0.4,
        'relative_humidity_2m': 0.3,
        'precipitation': -0.3,
        'wind_speed_10m': -0.1
    }
    tci_features = [feat for feat in tci_weights.keys() if feat in df.columns]
    if tci_features:
        scaler_mm = MinMaxScaler()
        df_norm = pd.DataFrame(scaler_mm.fit_transform(df[tci_features]),
                               columns=tci_features, index=df.index)
        df['travel_comfort_index'] = 0
        for feat in tci_features:
            df['travel_comfort_index'] += tci_weights[feat] * df_norm[feat]
    else:
        df['travel_comfort_index'] = np.nan

    if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
        df['temp_humidity_interaction'] = df['temperature_2m'] * df['relative_humidity_2m']

    # ---------------------------
    # Encode Categorical Variables (Country, city, season)
    # ---------------------------
    for cat in ['Country', 'city', 'season']:
        if cat in df.columns:
            df[cat] = df[cat].astype(str).str.strip()
            dummies = pd.get_dummies(df[cat], prefix=cat, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[cat], inplace=True)

    # ---------------------------
    # Create TCI Classes (Low, Medium, High)
    # ---------------------------
    df['TCI_class'] = pd.qcut(df['travel_comfort_index'], q=3, labels=['Low', 'Medium', 'High'])

    # ---------------------------
    # Scaling Numeric Features for Modeling
    # ---------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler_std = StandardScaler()
    df[numeric_cols] = scaler_std.fit_transform(df[numeric_cols])
    
    # At this point, we need to ensure that the final feature set used for prediction has exactly 65 columns.
    # We assume that the model was trained on 65 features (excluding the 'date' column and target column).
    # Let's remove the non-feature columns and then pad with zeros if necessary.
    features_df = df.drop(columns=["date", "TCI_class"], errors='ignore')
    current_feature_count = features_df.shape[1]
    if current_feature_count < 65:
        # Add dummy columns
        for i in range(65 - current_feature_count):
            features_df[f"dummy_{i}"] = 0
    elif current_feature_count > 65:
        st.error(f"Error: Processed feature count ({current_feature_count}) exceeds expected 65. Adjust your processing.")
    # Optionally, you can combine with the date if needed, but for model input we only use the 65 features.
    return pd.concat([df[["date"]], features_df], axis=1)

# --------------------------------------
# Main App Layout & User Inputs (Forecast & Historical Predictions)
# --------------------------------------
st.title("🌍 Smart Travel Planner")
st.subheader("Your AI-powered companion for personalized travel recommendations")

st.sidebar.header("Plan Your Trip")
selected_destination = st.sidebar.selectbox("Select your destination", list(destinations.keys()))
lat, lon = destinations[selected_destination]
location_query = f"{lat},{lon}"

# User selects forecast start date; forecast window = user date to user date + 7 days
user_forecast_start = st.sidebar.date_input("Select forecast start date", datetime.today() + timedelta(days=1))
forecast_end_date = user_forecast_start + timedelta(days=7)
preferred_climate = st.sidebar.selectbox("Preferred Weather Condition", ["Sunny", "Rainy", "Mild", "Cold"])

# Historical data settings: Last 7 days from current date
show_historical = st.sidebar.checkbox("Show Historical Weather Data (Last 7 Days)")
if show_historical:
    hist_start = datetime.today() - timedelta(days=7)
    hist_end = datetime.today()

# --------------------------------------
# Main Button: Get Forecast Predictions and Process for Prediction
# --------------------------------------
if st.sidebar.button("Get 7-Day Forecast Predictions"):
    # Fetch forecast data using WeatherAPI
    weather_data = get_weather_data(location_query, days=7)
    if weather_data:
        current_weather = weather_data.get("current", {})
        location_info = weather_data.get("location", {})
        forecast = weather_data.get("forecast", {}).get("forecastday", [])
        
        # Determine forecast available window
        forecast_api_start = datetime.strptime(forecast[0]["date"], "%Y-%m-%d").date()
        forecast_api_end = datetime.strptime(forecast[-1]["date"], "%Y-%m-%d").date()
        
        # Adjust user_forecast_start if needed
        if user_forecast_start < forecast_api_start:
            st.warning(f"Forecast is available from {forecast_api_start}. Adjusting start date.")
            user_forecast_start = forecast_api_start
        
        # Set forecast window: from user_forecast_start to the minimum of user_forecast_start+7 and forecast_api_end
        actual_forecast_end = min(user_forecast_start + timedelta(days=7), forecast_api_end)
        
        # Filter forecast data for days within the window
        filtered_forecast = [day for day in forecast if 
                             user_forecast_start <= datetime.strptime(day["date"], "%Y-%m-%d").date() <= actual_forecast_end]
        
        if not filtered_forecast:
            st.error("No forecast data available for the selected date range.")
        else:
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
            
            optimal_date = find_optimal_travel_date(filtered_forecast, user_forecast_start)
            st.markdown(f"### Optimal Travel Date Suggestion: **{optimal_date}**")
            
            forecast_df = pd.DataFrame([{
                "Date": f["date"],
                "Max Temp": f["day"]["maxtemp_c"],
                "Min Temp": f["day"]["mintemp_c"],
                "Avg Temp": f["day"]["avgtemp_c"],
                "Total Precipitation": f["day"].get("totalprecip_mm", 0),
                "Avg Humidity": f["day"].get("avghumidity", 0),
                "Max Wind": f["day"].get("maxwind_kph", 0)
            } for f in filtered_forecast])
            st.dataframe(forecast_df)
            fig = px.line(forecast_df, x="Date", y=["Max Temp", "Min Temp", "Avg Temp"],
                          title="Forecast Temperature Trend", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # --- 7-Day Travel Comfort Predictions from Forecast Data ---
            # After computing optimal_date, print it:
            st.markdown(f"### Optimal Travel Date Suggestion: **{optimal_date}**")

            # Find the forecast day corresponding to the optimal date
            optimal_day = next((day for day in filtered_forecast if day["date"] == optimal_date), None)

            if optimal_day:
                hourly_forecasts = optimal_day.get("hour", [])
                hourly_predictions = []
                for hour_data in hourly_forecasts:
                    # Construct feature vector from hourly data using keys from WeatherAPI.
                    # (Adjust keys if necessary. Here we use temp_c, precip_mm, wind_kph, and humidity.)
                    temp = hour_data.get("temp_c")
                    precip = hour_data.get("precip_mm", 0)
                    wind = hour_data.get("wind_kph", 0)
                    hum = hour_data.get("humidity", 0)
                    features = np.array([temp, precip, wind, hum])
                    expected_features = model.n_features_in_
                    # Pad with zeros if model expects more features than 4.
                    if expected_features > 4:
                        features = np.pad(features, (0, expected_features - 4), mode='constant')
                    features = features.reshape(1, -1)
                    pred = model.predict(features)
                    hourly_predictions.append(pred[0])
                
                # Count occurrences of each predicted label
                unique, counts = np.unique(hourly_predictions, return_counts=True)
                # Map the numeric predictions to labels using your comfort_map.
                comfort_map = {0: "Low", 1: "Medium", 2: "High"}
                prediction_summary = {comfort_map[k]: v for k, v in zip(unique, counts)}
                # Ensure all classes are represented
                for label in ["Low", "Medium", "High"]:
                    if label not in prediction_summary:
                        prediction_summary[label] = 0
                        
                st.markdown("### Hourly Prediction Summary for Optimal Date")
                st.write(prediction_summary)
            else:
                st.error("Optimal date not found in forecast data.")

            
            # --- Historical Data: Print Last 7 Days from Current Date ---
            if show_historical:
                df_hist = get_full_historical_dataframe(lat, lon, hist_start.strftime("%Y-%m-%d"), hist_end.strftime("%Y-%m-%d"))
                if df_hist is not None:
                    st.subheader("Historical Weather Data (Last 7 Days)")
                    st.dataframe(df_hist)
                    st.line_chart(df_hist.set_index("date")["temperature_2m"])
                else:
                    st.error("Failed to retrieve historical data.")
    else:
        st.error("Could not retrieve forecast data. Please try again.")
