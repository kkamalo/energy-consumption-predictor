import streamlit as st
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("pa-retp-indicators.csv")

st.title("Energy Consumption Predictor")

# User selects country and indicator
country = st.selectbox("Country", df["country"].unique())
indicator = st.selectbox("Indicator", df["indicator"].unique())

# Filter the data
data = df[(df["country"] == country) & (df["indicator"] == indicator)].sort_values("year")

st.write("Historical Data")
st.dataframe(data)

# Optional: Live data from World Bank
st.write("Live Data from World Bank")

code = country[:3].upper()
url = f"https://api.worldbank.org/v2/country/{code}/indicator/EG.USE.COMM.KT.OE?format=json"

try:
    response = requests.get(url).json()
    if isinstance(response, list) and len(response) > 1:
        recent = response[1][:5]
        live_data = [{"year": x["date"], "value": x["value"]} for x in recent if x["value"] is not None]
        st.write(pd.DataFrame(live_data))
    else:
        st.write("No data available.")
except:
    st.write("Couldn't fetch live data, but that's okay.")

# Prediction
if st.button("Forecast Next 5 Years"):
    if len(data) < 4:
        st.error("Not enough data to make predictions.")
    else:
        series = data["value"]
        model = ARIMA(series, order=(2, 1, 2))
        result = model.fit()
        forecast = result.forecast(steps=5)

        st.write("Forecast")
        st.write(forecast)

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(data["year"], series, label="Historical")
        future_years = list(range(int(data["year"].max()) + 1, int(data["year"].max()) + 6))
        plt.plot(future_years, forecast, label="Forecast")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend()
        st.pyplot(plt)

        # Explanation
        st.write("Prediction Notes")
        st.write(f"The data for {indicator} in {country} shows a continuing trend. Based on past values, "
                 "the forecast predicts similar behavior over the next 5 years. Factors like energy demand and "
                 "growth trends are likely influencing this pattern.")
