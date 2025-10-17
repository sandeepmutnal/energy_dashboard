from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Home Page - upload CSV
@app.route('/')
def index():
    return render_template('index.html')

# Upload and process CSV
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load data
    data = pd.read_csv(filepath)

    # Simple forecast using linear regression
    if 'Date' in data.columns and 'Consumption' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Day'] = np.arange(len(data))
        X = data[['Day']]
        y = data['Consumption']

        model = LinearRegression()
        model.fit(X, y)

        # Predict next 7 days
        future_days = np.arange(len(data), len(data) + 7).reshape(-1, 1)
        forecast = model.predict(future_days)

        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], y, label='Actual')
        plt.plot(pd.date_range(data['Date'].iloc[-1], periods=8, freq='D')[1:], forecast, label='Forecast', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Energy Consumption')
        plt.title('Energy Consumption Forecast')
        plt.legend()
        chart_path = os.path.join('static', 'forecast.png')
        plt.savefig(chart_path)
        plt.close()

        return render_template('dashboard.html', forecast=list(np.round(forecast, 2)), chart_path=chart_path)
    else:
        return "CSV must have 'Date' and 'Consumption' columns!"

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.mkdir('uploads')
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

