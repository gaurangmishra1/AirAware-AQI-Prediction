# ===========================================
# Milestone 4: Streamlit Dashboard
# AirAware - Smart AQI Prediction System
# ===========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

# ==============================
# Load Data
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("../Milestone1/city_day_cleaned.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_alerts():
    if os.path.exists("../Milestone3/alerts_trend_data.csv"):
        return pd.read_csv("../Milestone3/alerts_trend_data.csv")
    else:
        return None

df = load_data()
alerts_df = load_alerts()

# ==============================
# Load Best Model
# ==============================
best_model = None
model_type = None

if os.path.exists("../Milestone2/best_model.pkl"):
    best_model = joblib.load("../Milestone2/best_model.pkl")
    model_type = "ARIMA/Prophet"
elif os.path.exists("../Milestone2/best_model_lstm.h5"):
    best_model = load_model("../Milestone2/best_model_lstm.h5", compile=False)
    model_type = "LSTM"

# ==============================
# Streamlit App
# ==============================
st.set_page_config(page_title="AirAware AQI Prediction", layout="wide")

st.title("🌍 AirAware - Smart AQI Prediction System")
st.markdown("Infosys Springboard Internship Project - Milestone 4")

# Sidebar
st.sidebar.header("User Options")
cities = df["City"].unique()
selected_city = st.sidebar.selectbox("Select City", cities)
selected_feature = st.sidebar.selectbox("Select Feature for Trend", ["AQI","PM2.5","PM10","NO2","SO2","O3","CO"])

# Filter data
city_data = df[df["City"] == selected_city]

# ==============================
# Trend Visualization
# ==============================
st.subheader(f"📈 Trend Analysis for {selected_city}")

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(city_data["Date"], city_data[selected_feature], label=selected_feature, color="blue")
ax.set_title(f"{selected_feature} Trend in {selected_city}")
ax.set_xlabel("Date")
ax.set_ylabel(selected_feature)
ax.legend()
st.pyplot(fig)

# ==============================
# Alerts Visualization
# ==============================
st.subheader(f"⚠️ AQI Alerts for {selected_city}")
if alerts_df is not None:
    city_alerts = alerts_df[alerts_df["City"] == selected_city]["AQI_Category"].value_counts()
    st.bar_chart(city_alerts)
else:
    st.write("No alerts data available.")

# ==============================
# Prediction
# ==============================
st.subheader(f"🤖 AQI Prediction for {selected_city}")

if model_type == "ARIMA/Prophet":
    st.write("Using ARIMA/Prophet Model")
    forecast = best_model.forecast(steps=7)
    st.line_chart(forecast)

elif model_type == "LSTM":
    st.write("Using LSTM Model")
    # Prepare last 30 days for prediction
    scaler = MinMaxScaler(feature_range=(0,1))
    values = city_data["AQI"].dropna().values.reshape(-1,1)
    scaled = scaler.fit_transform(values)

    X_input = scaled[-30:]
    X_input = X_input.reshape(1, X_input.shape[0], 1)

    preds = []
    temp_input = X_input
    for i in range(7):  # predict next 7 days
        yhat = best_model.predict(temp_input, verbose=0)
        preds.append(yhat[0][0])
        yhat_reshaped = yhat.reshape(1,1,1)  # ensure same 3D shape
        temp_input = np.append(temp_input[:,1:,:], yhat_reshaped, axis=1)

    preds_rescaled = scaler.inverse_transform(np.array(preds).reshape(-1,1))

    future_dates = pd.date_range(city_data["Date"].max() + pd.Timedelta(days=1), periods=7)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_AQI": preds_rescaled.flatten()})

    st.line_chart(forecast_df.set_index("Date"))

else:
    st.write("No trained model found. Please run Milestone 2 first.")

# ==============================
# Correlation Heatmap
# ==============================
st.subheader(f"🔗 Correlation between Pollutants and AQI - {selected_city}")

plt.figure(figsize=(8,5))
sns.heatmap(city_data[["PM2.5","PM10","NO2","SO2","O3","CO","AQI"]].corr(),
            annot=True, cmap="coolwarm")
st.pyplot(plt)
