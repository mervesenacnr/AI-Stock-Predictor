import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import io
import base64
import json
from datetime import datetime

def predict_stock_with_lstm(symbol="THYAO.IS", start_date="2020-01-01", end_date="2025-01-01"):
    try:
        if pd.to_datetime(end_date) > datetime.today():
            return None

        df = yf.download(symbol, start=start_date, end=end_date)
        if df.empty or df.isnull().all().all():
            return None

        data = df[['Close']].values
        if len(data) < 100:
            return None

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        prediction_days = 60
        x_train, y_train = [], []
        for i in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[i - prediction_days:i])
            y_train.append(scaled_data[i])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

        predicted_train = model.predict(x_train)
        predicted_train_inv = scaler.inverse_transform(predicted_train)
        y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1))
        r2 = r2_score(y_train_true, predicted_train_inv)
        print(f"📊 Model R² Score: {r2:.4f}")

        last_sequence = scaled_data[-prediction_days:]
        last_sequence = np.reshape(last_sequence, (1, prediction_days, 1))
        prediction = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        real_price = data[-100:].flatten()
        last_day = df.index[-1]
        future_day = last_day + pd.Timedelta(days=1)

        # Plotly uyumlu JSON veri (go.Scatter yerine dict!)
        trace1 = {
            "x": [str(date) for date in df.index[-100:]],
            "y": real_price.tolist(),
            "type": "scatter",
            "mode": "lines",
            "name": "Real Price"
        }
        trace2 = {
            "x": [str(last_day), str(future_day)],
            "y": [float(real_price[-1]), float(predicted_price)],
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Predicted",
            "line": {"dash": "dash", "color": "red"}
        }
        layout = {
            "title": f"{symbol} Prediction",
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Price"}
        }
        plot_json = json.dumps({"data": [trace1, trace2], "layout": layout})

        # Statik görsel
        fig, ax = plt.subplots()
        ax.plot(df.index[-100:], real_price, label='Real Price')
        ax.plot([last_day, future_day], [real_price[-1], predicted_price],
                label='Predicted', linestyle='--', color='red', marker='o')
        ax.legend()
        ax.set_title(f"{symbol} 1-Day Prediction")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode()

        return image_base64, round(predicted_price, 2), plot_json

    except Exception as e:
        print("❌ Error in model:", e)
        return None
