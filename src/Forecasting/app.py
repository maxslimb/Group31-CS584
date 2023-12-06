# Import all the required packages
from flask import Flask, jsonify, request, make_response
import os
from pmdarima import auto_arima
from dateutil import *
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from flask_cors import CORS
from datetime import datetime as dt

# Tensorflow (Keras & LSTM) related packages
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Facebook Prophet packages
from werkzeug.http import is_resource_modified
import json
import dateutil.relativedelta
from dateutil import *
from datetime import date
import requests
import matplotlib.pyplot as plt
from prophet import Prophet 
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Stats Model Packages
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Import required storage package from Google Cloud Storage
from google.cloud import storage
matplotlib.use('agg')
# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)
# Initlize Google cloud storage client
client = storage.Client()

# Add response headers to accept all types of  requests

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

#  Modify response headers when returning to the origin

def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

@app.route('/api/statmis', methods=['POST'])
def statmis():
    body = request.get_json()
    type = body["type"]
    repo_name = body["repo"]
    print("type",type)
    issues = body["issues"]

    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')

    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    LOCAL_IMAGE_PATH = "static/images/"
    OBSERVATION_IMAGE_NAME = "stats_observation_" + type +"_"+ repo_name + ".png"
    OBSERVATION_IMAGE_URL = BASE_IMAGE_PATH + OBSERVATION_IMAGE_NAME

    FORECAST_IMAGE_NAME = "stats_forecast_" + type +"_" + repo_name + ".png"
    FORECAST_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME

    df = pd.DataFrame(issues)
    df1 = df.groupby(['created_at'], as_index = False).count()
    dataFrame = df1[['created_at','issue_number']]
    dataFrame.columns = ['ds', 'y']
    dataFrame.set_index('y')
    period = len(dataFrame) // 2
    predict = sm.tsa.seasonal_decompose(dataFrame.index, period=period)
    figure = predict.plot()
    figure.set_size_inches(12,7)
    plt.title("Observations plot of created issues")
    figure.get_figure().savefig(LOCAL_IMAGE_PATH + OBSERVATION_IMAGE_NAME)               #observation image
    model = sm.tsa.ARIMA(dataFrame['y'].iloc[1:], order = (1, 0, 0))
    results = model.fit()
    dataFrame['forecast'] = results.fittedvalues
    fig = dataFrame[['y', 'forecast']].plot(figsize=(12,7))
    plt.title("Timeseries forecasting of created issues")
    if len(dataFrame['forecast']) == len(dataFrame['y']):
        y_true = dataFrame['y'].iloc[1:]  # Skip the first value if necessary
        y_pred = dataFrame['forecast'].iloc[1:]
    else:
        # Assuming the forecast is one element shorter than the actual values
        y_true = dataFrame['y'].iloc[1:]  # Skip the first value to align the lengths
        y_pred = dataFrame['forecast']

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"Test Accuracy_StatModel MSE: {mse}, MAE: {mae}, RMSE: {rmse}")
    fig.get_figure().savefig(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)                 

     # Uploads an images into the google cloud storage bucket
    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(OBSERVATION_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + OBSERVATION_IMAGE_NAME)

    new_blob = bucket.blob(FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)

    # Construct the response
    json_response = {
        "stats_observation_url": OBSERVATION_IMAGE_URL,
        "stats_forecast_url": FORECAST_IMAGE_URL
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)


@app.route('/api/fbprophetis', methods=['POST'])
def fbprophetis():
    body = request.get_json()
    type = body["type"]
    repo_name = body["repo"]
    issues = body["issues"]

    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')

    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    LOCAL_IMAGE_PATH = "static/images/"
    FORECAST_IMAGE_NAME = "fbprophet_forecast_" + type +"_"+ repo_name + ".png"
    FORECAST_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME

    FORECAST_COMPONENTS_IMAGE_NAME = "fbprophet_forecast_components_" + type +"_" + repo_name + ".png"
    FORECAST_COMPONENTS_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME

    df = pd.DataFrame(issues)
    df1 = df.groupby(['created_at'], as_index = False).count()
    dataFrame = df1[['created_at','issue_number']]
    dataFrame.columns = ['ds', 'y']
    split_point = len(dataFrame) - 30  # Reserve last 30 days for testing
    train = dataFrame[:split_point]
    test = dataFrame[split_point:]
    model = Prophet(yearly_seasonality=True)
    model.fit(train)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    y_pred = forecast['yhat'][-30:]
    y_true = test['y']

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"Test Accuracy_Prophet MSE: {mse}, MAE: {mae}, RMSE: {rmse}")
    forcast_fig1 = model.plot(forecast)
    forcast_fig2 = model.plot_components(forecast)
    forcast_fig1.savefig(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)
    forcast_fig2.savefig(LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    

    # Uploads an images into the google cloud storage bucket
    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)

    new_blob = bucket.blob(FORECAST_COMPONENTS_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    # Construct the response
    json_response = {
        "fbprophet_forecast_url": FORECAST_IMAGE_URL,
        "fbprophet_forecast_components_url": FORECAST_COMPONENTS_IMAGE_URL
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)


@app.route('/api/forecast', methods=['POST'])
def forecast():
    body = request.get_json()
    issues = body["issues"]
    type = body["type"]
    repo_name = body["repo"]
    data_frame = pd.DataFrame(issues)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']
    print("df: ", df)
    df['ds'] = pd.to_datetime(df['ds'])

    # Converting to numpy array
    array = df.to_numpy()
    x = np.array([time.mktime(i[0].timetuple()) for i in array])
    y = np.array([i[1] for i in array])

    print("Y: ", y)

    # Find the first day directly from the DataFrame
    firstDay = df['ds'].min()

    '''
    To achieve data consistency with both actual data and predicted values, 
    add zeros to dates that do not have orders
    [firstDay + timedelta(days=day) for day in range((max(df['ds']) - firstDay).days + 1)]
    '''
    Ys = [0] * ((max(df['ds']) - firstDay).days + 1)
    days = pd.Series([firstDay + timedelta(days=i)
                    for i in range(len(Ys))])
    for d, y_val in zip(df['ds'], y):
        Ys[(d - firstDay).days] = y_val

    # Modify the data that is suitable for LSTM
    Ys = np.array(Ys)
    Ys = Ys.astype('float32')
    Ys = np.reshape(Ys, (-1, 1))
    # Apply min max scaler to transform the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys = scaler.fit_transform(Ys)
    # Divide training - test data with 80-20 split
    train_size = int(len(Ys) * 0.80)
    test_size = len(Ys) - train_size
    train, test = Ys[0:train_size, :], Ys[train_size:len(Ys), :]
    print('train size:', len(train), ", test size:", len(test))

    # Create the training and test dataset
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    '''
    Look back decides how many days of data the model looks at for prediction
    Here LSTM looks at approximately one month data
    '''
    look_back = min(30, len(test) - 2)
    print(len(test))

    # Assuming create_dataset is a function that formats the data for LSTM
    if len(test) > look_back + 1:
        X_test, Y_test = create_dataset(test, look_back)
    else:
        print("Test dataset is too small for the specified look_back period.")
    X_train, Y_train = create_dataset(train, look_back)
   # X_test, Y_test = create_dataset(test, look_back)

    # Additional check: Ensure that X_test and X_train are not empty
    if len(X_train) == 0:
        print("X_train is empty. Check the train dataset and look_back parameter.")
    if len(X_test) == 0:
        print("X_test is empty. Check the test dataset and look_back parameter.")

    # Reshape input to be [samples, time steps, features]
    # Add a check to avoid reshaping if the dataset is empty
    if len(X_train) > 0:
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    if len(X_test) > 0:
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Verifying the shapes
    print('Shapes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


    # Model to forecast
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    print("TEST ---------")
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))  # Output layer: adjust the number of neurons and activation function based on your task

    # Compile the model - this example is for a regression task
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'],run_eagerly=True)

    # Fit the model with training data and set appropriate hyper parameters
    history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print("Test Accuracy_LSTM: {:.2f}%".format(accuracy * 100))

    '''
    Creating image URL
    BASE_IMAGE_PATH refers to Google Cloud Storage Bucket URL.Add your Base Image Path in line 145
    if you want to run the application local
    LOCAL_IMAGE_PATH refers local directory where the figures generated by matplotlib are stored
    These locally stored images will then be uploaded to Google Cloud Storage
    '''
    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'Your_Base_Image_path')
    # DO NOT DELETE "static/images" FOLDER as it is used to store figures/images generated by matplotlib
    LOCAL_IMAGE_PATH = "static/images/"

    # Creating the image path for model loss, LSTM generated image and all issues data image
    MODEL_LOSS_IMAGE_NAME = "model_loss_" + type +"_"+ repo_name + ".png"
    MODEL_LOSS_URL = BASE_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME

    LSTM_GENERATED_IMAGE_NAME = "lstm_generated_data_" + type +"_" + repo_name + ".png"
    LSTM_GENERATED_URL = BASE_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME

    ALL_ISSUES_DATA_IMAGE_NAME = "all_issues_data_" + type + "_"+ repo_name + ".png"
    ALL_ISSUES_DATA_URL = BASE_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME

    DAY_MAX_ISSUE_CREATED_IMAGE_NAME = "day_max_issues_created_data_" + type + "_"+ repo_name + ".png"
    DAY_MAX_ISSUE_CREATED_DATA_URL = BASE_IMAGE_PATH + DAY_MAX_ISSUE_CREATED_IMAGE_NAME

    DAY_MAX_ISSUE_CLOSED_IMAGE_NAME = "day_max_issues_closed_data_" + type + "_"+ repo_name + ".png"
    DAY_MAX_ISSUE_CLOSED_DATA_URL = BASE_IMAGE_PATH + DAY_MAX_ISSUE_CLOSED_IMAGE_NAME

    MONTH_MAX_ISSUE_CLOSED_IMAGE_NAME = "month_max_issues_closed_data_" + type + "_"+ repo_name + ".png"
    MONTH_MAX_ISSUE_CLOSED_DATA_URL = BASE_IMAGE_PATH + MONTH_MAX_ISSUE_CLOSED_IMAGE_NAME


    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get(
        'BUCKET_NAME', 'Your_BUCKET_NAME')

    # Model summary()

    # Plot the model loss image
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss For ' + type)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)

    # Predict issues for test data
    y_pred = model.predict(X_test)

    # Plot the LSTM Generated image
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(np.arange(0, len(Y_train)), Y_train, 'g', label="history")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             Y_test, marker='.', label="true")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             y_pred, 'r', label="prediction")
    axs.legend()
    axs.set_title('LSTM Generated Data For ' + type)
    axs.set_xlabel('Time Steps')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)

    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('All Issues Data')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)

    # Uploads an images into the google cloud storage bucket
    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(MODEL_LOSS_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)
    new_blob = bucket.blob(ALL_ISSUES_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)
    new_blob = bucket.blob(LSTM_GENERATED_IMAGE_NAME)
    new_blob.upload_from_filename(
        filename=LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)
    new_blob = bucket.blob(DAY_MAX_ISSUE_CREATED_IMAGE_NAME)

    # Construct the response
    json_response = {
        "model_loss_image_url": MODEL_LOSS_URL,
        "lstm_generated_image_url": LSTM_GENERATED_URL,
        "all_issues_data_image": ALL_ISSUES_DATA_URL
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)


# Run LSTM app server on port 8080
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)