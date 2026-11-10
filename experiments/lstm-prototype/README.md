# AI Stock Predictor — LSTM Prototype

An experimental stock-price forecasting application developed
as part of my Computer Engineering graduation project.

## Features

- Historical stock data retrieval using Yahoo Finance
- User-selected stock symbol and date range
- Two-layer LSTM neural network (50 units per layer)
- 60-step lookback window and MinMax normalization
- Next-step closing-price forecast
- Matplotlib and Plotly-compatible chart generation
- Flask-based web interface

## Technology

Python, TensorFlow/Keras, scikit-learn, Flask,
pandas, NumPy, yfinance and Matplotlib.

## Running Locally

Use Python 3.10 with a compatible TensorFlow installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

Dependency compatibility and end-to-end execution should
be verified in the target environment.

## Research Limitations

The current implementation trains a new model for each
prediction request using five epochs.

Its reported R-squared value is calculated on training data,
not on an independent held-out test set. The scaler is also
fitted to the full downloaded series.

Therefore, this prototype does not establish reliable
out-of-sample forecasting performance. Future work includes
chronological validation, leakage-free preprocessing,
baselines, and reproducible performance evaluation.

This is an educational research prototype, not financial advice.
