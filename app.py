import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
st.set_page_config(page_title="DL Sales Forecast", layout="wide")

st.title("🧠 Deep Learning Based Sales Forecasting")
st.markdown("""
This application uses a Deep Neural Network to predict weekly sales based on store, time, and holiday factors.
""")

# ------------------ LOAD DATA ------------------
df = pd.read_csv("train.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# ------------------ LOAD MODEL ------------------
model = load_model("sales_model.keras")
scaler = joblib.load("scaler.save")
mae = joblib.load("mae.save")
history = joblib.load("history.save")
st.markdown("### 📊 Model Performance")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.caption("MAE represents the average error between predicted and actual sales. Lower values indicate better performance.")
# ------------------ SIDEBAR ------------------
st.sidebar.header("📥 Input Features")

store = st.sidebar.number_input("Store", min_value=1, value=1)
dept = st.sidebar.number_input("Department", min_value=1, value=1)
is_holiday = st.sidebar.selectbox("Holiday", [0, 1])

year = st.sidebar.number_input("Year", value=2012)
month = st.sidebar.slider("Month", 1, 12, 1)
day = st.sidebar.slider("Day", 1, 31, 1)

# ------------------ PREDICTION ------------------
if st.sidebar.button("Predict"):
    input_data = np.array([[store, dept, is_holiday, year, month, day]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.subheader("📊 Predicted Weekly Sales")
    st.success(f"{prediction[0][0]:,.2f}")

# ------------------ SECTION DIVIDER ------------------
st.markdown("---")
st.markdown("## 📊 Data Insights & Model Analysis")
# ------------------ GRAPH 1: Monthly Pattern ------------------
st.subheader("📈 Learning Seasonal Patterns (Month vs Sales)")

monthly_sales = df.groupby("Month")["Weekly_Sales"].mean()

fig1, ax1 = plt.subplots()
ax1.plot(monthly_sales.index, monthly_sales.values)
ax1.set_title("Model learns seasonal behavior")
ax1.set_xlabel("Month")
ax1.set_ylabel("Avg Weekly Sales")

st.pyplot(fig1)
st.caption("This graph shows seasonal trends in sales. The model learns how sales change across different months.")
# ------------------ GRAPH 2: Holiday Impact ------------------
st.subheader("🎯 Learned Holiday Impact")

holiday_sales = df.groupby("IsHoliday")["Weekly_Sales"].mean()

fig2, ax2 = plt.subplots()
ax2.bar(["No Holiday", "Holiday"], holiday_sales.values)
ax2.set_title("Model captures holiday spikes")
ax2.set_ylabel("Avg Weekly Sales")

st.pyplot(fig2)
st.caption("This graph shows that sales increase during holidays, which the model uses as an important feature.")

# ------------------ GRAPH 3: Prediction vs Actual ------------------
st.subheader("📉 Model Behavior (Actual vs Predicted Sample)")

# Take small sample
sample = df.sample(100)

X_sample = sample[["Store", "Dept", "IsHoliday", "Year", "Month", "Day"]]
y_actual = sample["Weekly_Sales"]

X_scaled_sample = scaler.transform(X_sample)
y_pred = model.predict(X_scaled_sample)

fig3, ax3 = plt.subplots()
ax3.scatter(y_actual, y_pred)
ax3.set_xlabel("Actual Sales")
ax3.set_ylabel("Predicted Sales")
ax3.set_title("Prediction Accuracy Visualization")

st.pyplot(fig3)
st.caption("Each point represents a prediction. Closer alignment between actual and predicted values indicates better model performance.")
st.subheader("📉 Training Loss Curve")

fig, ax = plt.subplots()

ax.plot(history['loss'], label='Training Loss')
ax.plot(history['val_loss'], label='Validation Loss')

ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Model Learning Curve")
ax.legend()

st.pyplot(fig)
st.caption("This graph shows how the model improves during training. Decreasing loss indicates effective learning.")
