""" Implement price forecasting for a stock """

import os
import numpy as np
import pandas as pd
from flask import render_template
from markupsafe import Markup
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree, metrics

from indicators import plot_stock_prices_prediction, get_momentum, get_rolling_mean, get_rolling_std, get_RSI, get_sma, \
    get_bollinger_bands
from util import symbol_to_path, fetchOnlineData, get_data
import datetime as dt

def showforcastpricesvalues(request):

    # Get symbol
    symbol = str(request.get('ticker_select'))

    # Get Forecast date
    forecast_date = str(request.get('forecastDate'))

    # Get Forecast model
    forecast_model = str(request.get('model_Selection'))

    # Get Forecast time
    forecast_time = int(request.get('forecast_Time'))

    # Get Lookback
    forecast_lookback = int(request.get('look_Back'))


    # Get 1 year of data to train and test
    start_d = dt.date.today() - dt.timedelta(forecast_lookback)
    yesterday = dt.date.today() - dt.timedelta(1)

    # Check whether there is a file with input data or not before dowunloading
    file = symbol_to_path(symbol)
    if os.path.isfile(file):
        (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(file)
        file_date = dt.datetime.utcfromtimestamp(ctime).strftime('%Y-%m-%d')
        today = dt.date.today()
        if file_date != str(today):
        # Get dates from initial date to yesterday from Yahoo
            try:
                download = fetchOnlineData(start_d, symbol)
                if download == False:
                    return render_template(
                        # name of template
                        "pricesForecastForm.html",
                        error=True)
            except:
                return render_template(
                    # name of template
                    "pricesForecastForm.html",
                    error=True)

    portf_value = get_data(symbol)

    # Normalize the prices Dataframe
    normed = portf_value.copy()
    #normed = scaling_data(normed, symbol)

    normed['date'] = portf_value.index
    normed.set_index('date', inplace=True)

    # ****Momentum chart****
    # Compute momentum
    sym_mom = get_momentum(normed[symbol], window=10)

    # ****Bollinger Bands****
    # Compute rolling mean
    rm_JPM = get_rolling_mean(portf_value[symbol], window=10)

    # Compute rolling standard deviation
    rstd_JPM = get_rolling_std(portf_value[symbol], window=10)

    # Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_JPM, rstd_JPM)

    # ****Relative Strength Index (RSI)****
    # Compute RSI
    rsi_value = get_RSI(portf_value[symbol])

    # ****Simple moving average (SMA)****
    # Compute SMA
    sma, q = get_sma(normed[symbol], window=10)

    # Create momentum column
    normed['Momentum'] = sym_mom

    # Create SMA column
    normed['SMA'] = sma

    # Create SMA column
    normed['RSI'] = rsi_value

    # Clean nan values
    normed = normed.fillna(0)

    # Define X and y
    feature_cols = ['Momentum', 'SMA', 'RSI']
    X = normed[feature_cols]
    y = normed[symbol]

    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


    # Loop through a few different max depths and check the performance
    for d in [3, 5, 10]:
        # Create the tree and fit it
        decision_tree = DecisionTreeRegressor(max_depth=d)
        decision_tree.fit(X_train, y_train)

        # Print out the scores on train and test
        print('max_depth=', str(d))
        print(decision_tree.score(X_train, y_train))
        print(decision_tree.score(X_test, y_test), '\n')

    model = tree.DecisionTreeRegressor(max_depth=10)

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Measuring predictions
    # Coefficient of determination R^2
    '''
    The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
    '''
    coef_deter = model.score(X_train, y_train)
    print('Coefficient of determination R^2: %s') %  coef_deter


    # Forecast error
    '''
    The units of the forecast error are the same as the units of the prediction. A forecast error of zero indicates no error, or perfect skill for that forecast.
    '''
    forecast_errors = [y_test[i] - y_pred[i] for i in range(len(y_test))]
    print('Forecast Errors: %s' % forecast_errors)

    # Forecast bias
    '''
    Mean forecast error, also known as the forecast bias. A forecast bias of zero, or a very small number near zero, shows an unbiased model.
    '''
    bias = sum(forecast_errors) * 1.0 / len(y_test)
    print('Bias: %f' % bias)

    # Mean absolute error
    '''
    A mean absolute error of zero indicates no error.
    '''
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

    # Mean squared error
    '''
    A mean squared error of zero indicates perfect skill, or no error.
    '''
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

    # Root mean squared error
    '''
    As with the mean squared error, an RMSE of zero indicates no error.
    '''
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




    results = pd.DataFrame({'Price': y_test, 'Price prediction': y_pred})
    results.sort_index(inplace=True)

    # Plot prediction

    plot_prices_pred = plot_stock_prices_prediction(results.index, results['Price'], results['Price prediction'])
    return symbol, start_d, yesterday, plot_prices_pred
