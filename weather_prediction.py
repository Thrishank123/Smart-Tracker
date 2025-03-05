import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Global list to store the country names (preserving order)
GLOBAL_COUNTRY_NAMES = []

def fetch_weather_data(start_date):
    global GLOBAL_COUNTRY_NAMES  # Use the global variable
    # Lists of countries, latitudes, and longitudes
    country = ["Bali", "Bangkok", "Barcelona", "Berlin", "Cape Town", "Dubai", "Istanbul", 
               "Lima", "London", "Los Angeles", "Melbourne", "New York City", "Paris", "Prague", 
               "Queenstown", "Rio De Janeiro", "Rome", "Seoul", "Singapore", "Sydney", "Tokyo", "Vienna"]
    latitude = ["-8.34", "13.73", "41.38", "52.52", "-33.91", "25.06", "41.16", "-12.04", 
                "51.51", "34.05", "-37.81", "40.71", "48.85", "50.08", "-45.03", "-22.90", 
                "41.90", "37.56", "1.36", "-33.86", "35.68", "48.20"]
    longitude = ["115.09", "100.52", "2.15", "13.41", "18.42", "55.17", "28.76", "-77.04", 
                 "-0.11", "-118.24", "144.96", "-74.00", "2.35", "14.42", "168.66", "-43.17", 
                 "12.49", "126.97", "103.8", "151.20", "139.69", "16.37"]

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    # Define full list of hourly weather variables (order is important)
    hourly_vars = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
        "precipitation_probability", "precipitation", "rain", "showers", "snowfall", "snow_depth",
        "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low",
        "cloud_cover_mid", "cloud_cover_high", "visibility", "evapotranspiration",
        "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m", "wind_speed_80m",
        "wind_speed_120m", "wind_speed_180m", "wind_direction_10m", "wind_direction_80m",
        "wind_direction_120m", "wind_direction_180m", "wind_gusts_10m", "temperature_80m",
        "temperature_120m", "temperature_180m", "soil_temperature_0cm", "soil_temperature_6cm",
        "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm",
        "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm",
        "soil_moisture_27_to_81cm"
    ]
    
    data_all = pd.DataFrame()
    url = "https://api.open-meteo.com/v1/forecast"
    
    for lat, lon, country_name in zip(latitude, longitude, country):
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "hourly": hourly_vars,
            "start_date": start_date,
            "end_date": start_date
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        
        # Create hourly time index using the provided start, end and interval
        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}
        # Assign each variable from the API response (order matters)
        for i, var in enumerate(hourly_vars):
            hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
        
        temp = pd.DataFrame(data=hourly_data)
        # Append the country name for each row into the global list
        GLOBAL_COUNTRY_NAMES.extend([country_name] * len(temp))
        # Keep the "Country" column for now (it will be dropped later during preprocessing)
        temp["Country"] = country_name
        
        data_all = pd.concat([data_all, temp])
        
    return data_all

def get_season(row):
    month = row['month']
    country_str = str(row['Country']).strip().lower() if 'Country' in row else ""
    # List of known southern hemisphere countries (adjust as needed)
    southern_countries = ['australia', 'new zealand', 'south africa', 'argentina', 'chile', 'brazil']
    if country_str in southern_countries:
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

def preprocess_data(data):
    # 1. Date Conversion and Time-Based Features
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['day_of_week'] = data['date'].dt.dayofweek   # Monday=0, Sunday=6
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year

    # 2. Define Season Based on Country and Month
    data['season'] = data.apply(get_season, axis=1)

    # 3. Handle Missing Data using ffill() and bfill()
    data.ffill(inplace=True)
    data.bfill(inplace=True)

    # 4. Drop Less Relevant Columns (soil-related columns)
    soil_columns = [col for col in data.columns if "soil" in col.lower()]
    if soil_columns:
        data.drop(columns=soil_columns, inplace=True)

    # 5. Advanced Feature Engineering
    # (a) Aggregated Wind Speed
    wind_cols = [col for col in ['wind_speed_10m', 'wind_speed_80m', 'wind_speed_120m', 'wind_speed_180m'] if col in data.columns]
    if wind_cols:
        data['avg_wind_speed'] = data[wind_cols].mean(axis=1)
    else:
        data['avg_wind_speed'] = np.nan

    # (b) Aggregated Cloud Cover
    cloud_cols = [col for col in ['cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high'] if col in data.columns]
    if cloud_cols:
        data['avg_cloud_cover'] = data[cloud_cols].mean(axis=1)

    # (c) Temperature Difference
    if 'apparent_temperature' in data.columns and 'temperature_2m' in data.columns:
        data['temp_difference'] = data['apparent_temperature'] - data['temperature_2m']

    # (d) Weather Severity Index
    severity_features = [col for col in ['precipitation', 'snowfall', 'showers'] if col in data.columns]
    if severity_features:
        data['weather_severity_index'] = data[severity_features].sum(axis=1)
    else:
        data['weather_severity_index'] = 0

    # (e) Travel Comfort Index (TCI)
    tci_weights = {
        'temperature_2m': 0.4,
        'relative_humidity_2m': 0.3,
        'precipitation': -0.3,
        'wind_speed_10m': -0.1
    }
    tci_features = [feat for feat in tci_weights.keys() if feat in data.columns]
    if tci_features:
        scaler_mm = MinMaxScaler()
        data_norm = pd.DataFrame(scaler_mm.fit_transform(data[tci_features]), 
                                 columns=tci_features, index=data.index)
        data['travel_comfort_index'] = 0
        for feat in tci_features:
            data['travel_comfort_index'] += tci_weights[feat] * data_norm[feat]
    else:
        data['travel_comfort_index'] = np.nan

    # (f) Interaction Feature: Temperature x Humidity
    if 'temperature_2m' in data.columns and 'relative_humidity_2m' in data.columns:
        data['temp_humidity_interaction'] = data['temperature_2m'] * data['relative_humidity_2m']

    # 6. Encode Categorical Variables (for 'Country', 'city', and 'season')
    for cat in ['Country', 'city', 'season']:
        if cat in data.columns:
            data[cat] = data[cat].astype(str).str.strip()
            dummies = pd.get_dummies(data[cat], prefix=cat, drop_first=True)
            data = pd.concat([data, dummies], axis=1)
            data.drop(columns=[cat], inplace=True)

    # 6b. Ensure expected dummy columns for "season" exist.
    expected_season_dummies = ['season_Summer', 'season_Winter']
    for col in expected_season_dummies:
        if col not in data.columns:
            data[col] = 0

    # 7. Create TCI Classes (Low, Medium, High)
    data['TCI_class'] = pd.qcut(data['travel_comfort_index'], q=3, labels=['Low', 'Medium', 'High'])

    # 8. Scale Numeric Features for Modeling
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    scaler_std = StandardScaler()
    data[numeric_cols] = scaler_std.fit_transform(data[numeric_cols])
    
    return data

def predict_weather(start_date):
    """
    Given a start_date (YYYY-MM-DD), this function:
      1. Fetches weather data for all cities for the date.
      2. Preprocesses the data using your pipeline.
      3. Loads the pre-trained KNN model from "Knn.pkl".
      4. Returns a list of tuples (date, country, prediction label) for each hour.
    """
    data = fetch_weather_data(start_date)
    processed_data = preprocess_data(data)
    
    # Drop columns that are not used as features (including "date" and "Country")
    X = processed_data.drop(columns=["TCI_class", "date", "Country"], errors='ignore')
    
    # Load pre-trained KNN model (warning about version differences may appear)
    model = joblib.load("Knn.pkl")
    predictions = model.predict(X)
    
    # Convert numerical predictions to corresponding string labels using a mapping.
    # Adjust the mapping if your training used a different encoding.
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    predictions_str = [label_map.get(p, p) for p in predictions]
    
    # Use the global country names for output.
    result = list(zip(processed_data['date'], GLOBAL_COUNTRY_NAMES, predictions_str))
    return result
