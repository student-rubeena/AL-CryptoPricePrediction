# AL-CryptoPricePrediction
This project aims to develop an AI-powered price prediction system for cryptocurrency markets using historical data from the Binance exchange.
The system leverages an LSTM (Long Short-Term Memory) neural network to predict future price movements and provides performance metrics through backtesting.

Features
Data Collection: Fetches historical BTC/USDT price data from Binance API.
Data Preprocessing: Handles missing values, adds technical indicators (SMA and RSI), and normalizes data.
AI Model Implementation: Builds and trains an LSTM model for price prediction.
Backtesting Framework: Validates the model by comparing predictions with actual historical data.
Visualization: Plots predicted vs actual prices for performance assessment.



Setup Instructions
Prerequisites
Python 3.8 or later
Binance API Key and Secret (Read-only permission recommended)


Required Libraries
Install the required Python libraries using the following command:
pip install pandas numpy matplotlib scikit-learn tensorflow binance



Running the Project
Clone the repository:
git clone https://github.com/your-repo/crypto-price-predictor.git
Navigate to the project directory:
cd crypto-price-predictor
Replace API_KEY and API_SECRET in the script with your Binance API credentials.



Run the script:
python crypto_price_predictor.py


Explanation of Components

1. Data Collection

Historical candlestick data for BTC/USDT is fetched using Binance's API.

Data is structured into a Pandas DataFrame with relevant columns.

2. Data Preprocessing

Converts timestamps to datetime and sets them as the index.

Computes technical indicators:

SMA (Simple Moving Average): Provides a smoothed trend of prices.

RSI (Relative Strength Index): Indicates potential overbought or oversold conditions.

Normalizes data using MinMaxScaler to improve model performance.

3. LSTM Model Implementation

Uses a two-layer stacked LSTM architecture:

First LSTM layer with 50 units and return_sequences=True.

Second LSTM layer with 50 units.

A Dense layer to output a single predicted price.

Compiles the model using the Adam optimizer and Mean Squared Error (MSE) loss function.

Trains the model for 50 epochs with a batch size of 32.

4. Backtesting Framework

Predicts future prices using the test set.

Compares predicted prices with actual prices.

Calculates directional accuracy, i.e., how often the model correctly predicts the direction of price movement.

Displays a line plot comparing predicted and actual prices.


Example Output
Backtesting Accuracy: The percentage of correct directional predictions.
Visualization: A plot showing actual vs predicted prices.
Future Improvements
Integrate additional technical indicators.
Experiment with other AI models (e.g., GRU, Transformer-based models).
Enhance backtesting by simulating trades and calculating profit/loss.



License
This project is licensed under the MIT License.


Contact
For any questions or feedback, please contact rubeenamaddarki85@gmail.com.

Crypto predictor UI :
![Screenshot (18)](https://github.com/user-attachments/assets/a36332cd-d2a5-406a-977c-5758b5b71bfa)
