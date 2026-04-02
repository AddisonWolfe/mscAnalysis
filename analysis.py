import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np

load_dotenv()
API_KEY = os.getenv("API_KEY")
SYMBOL = os.getenv("SYMBOL")
BASE_URL = os.getenv("BASE_URL")


def get_stock_data():
    url = BASE_URL + "query?function=TIME_SERIES_WEEKLY&symbol=" + SYMBOL + "&outputsize=compact&apikey=" + API_KEY
    data = requests.get(url).json()
    df = pd.DataFrame.from_dict(data["Weekly Time Series"], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df.columns = ["open", "high", "low", "close", "volume"]

    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": int
    })

    df.to_csv("stock_data.csv", index=True)

def get_weather_data():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 19.8987, # Hawaii
        "longitude": 155.6659, 
        "start_date": "1999-11-12",
        "end_date": "2026-03-20",
        "daily": ["temperature_2m_max", "rain_sum", "wind_speed_10m_max"],
    }
    responses = openmeteo.weather_api(url, params = params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(1).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}

    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["rain_sum"] = daily_rain_sum
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max

    daily_dataframe = pd.DataFrame(data = daily_data)
    daily_dataframe.columns = ["date", "temperature_2m_max", "rain_sum", "wind_speed_10m_max"]
    daily_dataframe.to_csv("weather_data.csv", index=False)

def convert_weather_data():
    df = pd.read_csv("weather_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    weekly = df.resample("7D").agg({
        "temperature_2m_max": "mean",
        "rain_sum": "sum",
        "wind_speed_10m_max": "mean"
    }).reset_index()


    weekly.to_csv("weekly_weather.csv", index=False)

def temp_vs_price():
    model = LinearRegression()

    stock_df = pd.read_csv("stock_data.csv")
    weather_df = pd.read_csv("weekly_weather.csv")

    X = stock_df[["close"]]   # keep 2D for sklearn
    y = weather_df["rain_sum"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # make both arrays 1D for plotting
    X_test_1d = X_test["close"].to_numpy()
    y_pred_1d = np.asarray(y_pred).reshape(-1)

    sort_idx = np.argsort(X_test_1d)
    X_test_sorted = X_test_1d[sort_idx]
    y_pred_sorted = y_pred_1d[sort_idx]

    plt.figure(figsize=(8, 5))

    plt.scatter(X_train["close"], y_train, label="Training points")
    plt.scatter(X_test["close"], y_test, label="Test points")
    plt.plot(X_test_sorted, y_pred_sorted, label="Model line")

    plt.xlabel("Price")
    plt.ylabel("Rain")
    plt.title("Linear Regression: Price vs Rain")
    plt.legend()
    plt.show()

def prec_vs_price():
    model = LinearRegression()

    stock_df = pd.read_csv("stock_data.csv")
    weather_df = pd.read_csv("weekly_weather.csv")

    X = stock_df[["close"]]   # keep 2D for sklearn
    y = weather_df["temperature_2m_max"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # make both arrays 1D for plotting
    X_test_1d = X_test["close"].to_numpy()
    y_pred_1d = np.asarray(y_pred).reshape(-1)

    sort_idx = np.argsort(X_test_1d)
    X_test_sorted = X_test_1d[sort_idx]
    y_pred_sorted = y_pred_1d[sort_idx]

    plt.figure(figsize=(8, 5))

    plt.scatter(X_train["close"], y_train, label="Training points")
    plt.scatter(X_test["close"], y_test, label="Test points")
    plt.plot(X_test_sorted, y_pred_sorted, label="Model line")

    plt.xlabel("Price")
    plt.ylabel("Rain")
    plt.title("Linear Regression: Price vs Rain")
    plt.legend()
    plt.show()

def wind_vs_price():
    model = LinearRegression()

    stock_df = pd.read_csv("stock_data.csv")
    weather_df = pd.read_csv("weekly_weather.csv")

    #250ish weeks of data, we have no metric to account for the
    #average overall increase of the market overtime, so this should
    #reduce noise for the time being.
    #Eventually we should gather data to account for this increase however. 
    stock_df = stock_df[1100:1376].copy()
    weather_df = weather_df[1100:1376].copy()

    #volatility is our performance metric
    stock_df["performance"] = (
        (stock_df["high"] - stock_df["low"]) / stock_df["open"]
    )

    #lagging features
    #Does last weeks volatility, and weather effect this weeks performance?
    weather_df["lag_volatility_1"] = stock_df["performance"].shift(1)
    weather_df["lag_temp_1"] = weather_df["temperature_2m_max"].shift(1)
    weather_df["lag_wind_1"] = weather_df["wind_speed_10m_max"].shift(1)
    weather_df["lag_rain_1"] = weather_df["rain_sum"].shift(1)

    #drop row 1 for nans
    weather_df = weather_df.iloc[1:]
    stock_df = stock_df.iloc[1:]

    X = weather_df[["wind_speed_10m_max", "rain_sum", "temperature_2m_max", "lag_volatility_1", "lag_temp_1", "lag_wind_1", "lag_rain_1"]]
    y = stock_df["performance"]

    split_idx = int(len(X) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual Weekly Volatility")
    plt.ylabel("Predicted Weekly Volatility")
    plt.title("Actual vs Predicted Weekly Stock Performance")
    plt.show()

def main():
    wind_vs_price()

if __name__ == "__main__":
    main()

