import streamlit as st
import pandas as pd
import requests
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("GENAI_API_KEY")

client = genai.Client(api_key=api_key)

# Load dataset
df = pd.read_csv("pa-retp-indicators.csv")

st.title("Energy Consumption Predictor")

# Multi-country selection
countries = st.multiselect(
    "Select Countries to Compare", df["country"].unique(),
    default=[df["country"].iloc[0]]
)
indicator = st.selectbox("Indicator", df["indicator"].unique())

# Filter data for selected countries
data_filtered = df[(df["country"].isin(countries)) & (df["indicator"] == indicator)]

st.write("Historical Data")
st.dataframe(data_filtered)

# Optional: Live data from World Bank (latest 5 years)
st.write("Live Data from World Bank")
for country in countries:
    code = country[:3].upper()
    url = f"https://api.worldbank.org/v2/country/{code}/indicator/EG.USE.COMM.KT.OE?format=json"
    try:
        response = requests.get(url).json()
        if isinstance(response, list) and len(response) > 1:
            recent = response[1][:5]
            live_data = [{"year": x["date"], "value": x["value"]} for x in recent if x["value"] is not None]
            st.write(f"Live data for {country}")
            st.write(pd.DataFrame(live_data))
        else:
            st.write(f"No live data available for {country}")
    except:
        st.write(f"Couldn't fetch live data for {country}, but that's okay.")

# Forecast Next 5 Years
if st.button("Forecast Next 5 Years"):
    plt.figure(figsize=(12, 5))
    future_forecasts = {}

    for country in countries:
        country_data = data_filtered[data_filtered["country"] == country].sort_values("year")
        if len(country_data) < 4:
            st.warning(f"Not enough data for {country}")
            continue

        series = country_data["value"]
        model = ARIMA(series, order=(2, 1, 2))
        result = model.fit()
        forecast = result.forecast(steps=5)
        future_forecasts[country] = forecast

        # Plot historical + forecast
        plt.plot(country_data["year"], series, label=f"{country} Historical")
        future_years = list(range(int(country_data["year"].max()) + 1, int(country_data["year"].max()) + 6))
        plt.plot(future_years, forecast, '--', label=f"{country} Forecast")

        # Show forecast table
        st.write(f"Forecast for {country}")
        st.write(forecast)

    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title(f"{indicator} Comparison & Forecast")
    plt.legend()
    st.pyplot(plt)

# Generate AI Insights
if st.button("Generate AI Insights"):
    insights_text = ""
    for country in countries:
        country_data = data_filtered[data_filtered["country"] == country].sort_values("year")
        hist_data_text = country_data.to_string()

        prompt = f"""
        You are an energy data analyst.
        Here is the historical data for {country}:
        {hist_data_text}

        Provide a clear, short analysis including:
        - Trend summary
        - Whether energy use is rising or falling
        - Any unusual points
        - A simple recommendation
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        )
        insights_text += f"### Insights for {country}\n{response.text}\n\n"

    st.write("### AI Insights (Powered by Gemini)")
    st.markdown(insights_text)

