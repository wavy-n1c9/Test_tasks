from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import numpy
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import statsmodels
import pickle
import matplotlib.pyplot as plt
from io import BytesIO

img_buffer = None
model = pickle.load(open("model.pkl", "rb"))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global img_buffer
    # period = int(request.form['period'])
    fromtime= pd.to_datetime(request.form['start_year']+'-'+request.form['quartal_start'])
    endtime = pd.to_datetime(request.form['end_year'] + '-' + request.form['quartal_end'])
    if fromtime>endtime:
        (fromtime,endtime)=(endtime,fromtime)

    df = pd.read_csv('df.csv')
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index("Unnamed: 0", inplace=True)
    df.index.freq = 'QS-OCT'

    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    # forecast = model.forecast(steps=period)
    forecast = model.predict(start=fromtime, end=endtime)

    plt.figure(figsize=(20, 6))
    plt.plot(train.index, train.value.values, label='Train')
    plt.plot(test.index, test.value.values, label='Test')
    plt.plot(forecast.index,forecast.values, label='Forecast')
    plt.legend()
    plt.title('E-Commerce Retail Sales')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.tight_layout()

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    plt.close()

    forecast = dict(zip(forecast.index.strftime('%Y-%m-%d'), forecast.values))
    df = dict(zip(df.index.strftime('%Y-%m-%d'), df.value.values))
    return render_template('index.html', forecast=forecast, data=df, image_path='/plot')

@app.route('/plot')
def plot():
    return send_file(BytesIO(img_buffer.getvalue()), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)