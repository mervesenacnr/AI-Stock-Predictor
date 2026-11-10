from flask import Flask, render_template, request
from model import predict_stock_with_lstm

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    image = None
    predicted_value = None
    plot_json = None

    if request.method == 'POST':
        symbol = request.form.get('symbol', 'THYAO.IS')
        start_date = request.form.get('start_date', '2020-01-01')
        end_date = request.form.get('end_date', '2025-01-01')

        result = predict_stock_with_lstm(symbol, start_date, end_date)
        if result:
            image, predicted_value, plot_json = result

    return render_template('index.html',
                           image=image,
                           predicted_value=predicted_value,
                           plot_json=plot_json)

if __name__ == '__main__':
    app.run(debug=True)
