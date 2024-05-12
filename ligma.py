import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time


# Daten laden (hier Platzhalter, bitte ersetze 'PATH_TO_YOUR_CSV' mit dem Pfad zu deiner CSV-Datei)
ticker = ["GME"]
years = 10
start_date = (datetime.today() - relativedelta(years=years, days=1)).strftime('%Y-%m-%d')
end_date = (datetime.today() - relativedelta(days=1)).strftime('%Y-%m-%d')
X = yf.download(ticker, start=start_date, end=end_date, interval="1d")
df = X

# Wähle 'Close' als Zielvariable
close_prices = df['Close'].values
close_prices = close_prices.reshape(-1, 1)

# Daten normalisieren
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices_scaled = scaler.fit_transform(close_prices)

# Definiere, wie viele Tage verwendet werden, um die Vorhersage zu treffen
look_back = 50
X, Y = [], []
for i in range(look_back, len(close_prices_scaled)):
    X.append(close_prices_scaled[i-look_back:i, 0])
    Y.append(close_prices_scaled[i, 0])

X = np.array(X)
Y = np.array(Y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, Y, epochs=100, batch_size=32)

predicted_stock_price = model.predict(X)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualisieren der Ergebnisse
plt.plot(close_prices, color='blue', label='Original Aktienpreis')
plt.plot(predicted_stock_price, color='red', label='Vorhergesagter Aktienpreis')
plt.title('Aktienpreis Vorhersage')
plt.xlabel('Zeit')
plt.ylabel('Aktienpreis')
plt.legend()
plt.show()

# Aufteilung der Daten in Trainings- und Testsets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
train_X, test_X = X[:train_size], X[train_size:]
train_Y, test_Y = Y[:train_size], Y[train_size:]

# Trainiere das Modell mit Trainingsdaten
model.fit(train_X, train_Y, epochs=100, batch_size=32, validation_data=(test_X, test_Y))

