from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load dataset
df = pd.read_csv("data/walmart_sales.csv/train.csv")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df.drop(['Date'], axis=1, inplace=True)

# Features and target
X = df.drop("Weekly_Sales", axis=1)
y = df["Weekly_Sales"]

# Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "models/scaler.save")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build Model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
history=model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
# Predict on test data
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print("Model MAE:", mae)

# Save MAE
joblib.dump(mae, "models/mae.save")

# Save training history
joblib.dump(history.history, "models/history.save")

# Save model
model.save("models/sales_model.h5")

print("Training Completed Successfully.")