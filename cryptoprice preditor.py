import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Binance API keys (replace with your own keys)
API_KEY = 'your_api_key_here'
API_SECRET = 'your_api_secret_here'

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Fetch historical candlestick data for BTC/USDT (1-hour interval)
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 Jan 2021", "1 Jan 2025")

# Convert to DataFrame
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
    'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
])

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Convert relevant columns to float
df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

# Handle missing values (if any)
df = df.dropna()

# Feature engineering: Adding Simple Moving Average (SMA) and Relative Strength Index (RSI)
df['SMA_20'] = df['close'].rolling(window=20).mean()

def compute_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = compute_rsi(df['close'])

# Drop rows with NaN values (resulting from moving averages and RSI)
df = df.dropna()

# Normalize data using MinMaxScaler
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df)

# Convert back to DataFrame for easier manipulation
scaled_df = pd.DataFrame(scaled_df, columns=df.columns, index=df.index)

print("Data collection and preprocessing complete!")

# Save the preprocessed data to a CSV file
scaled_df.to_csv('preprocessed_crypto_data.csv')

# Prepare data for LSTM model
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 3])  # Predict closing price
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_df.values)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(1)  # Predict a single value (closing price)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Save the trained model
model.save('crypto_price_predictor_lstm.h5')

print("LSTM model training complete and saved!")

# Backtesting
predictions = model.predict(X_test)
predicted_prices = scaler.inverse_transform(
    np.concatenate((X_test[:, -1, :-1], predictions), axis=1))[:, -1]
actual_prices = scaler.inverse_transform(
    np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1))[:, -1]

# Calculate accuracy (directional)
correct_direction = np.sum((predicted_prices[1:] - predicted_prices[:-1]) * (actual_prices[1:] - actual_prices[:-1]) > 0)
accuracy = correct_direction / (len(predicted_prices) - 1) * 100
print(f"Backtesting accuracy: {accuracy:.2f}%")

# Plot predictions vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('Backtesting: Actual vs Predicted Prices')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()

