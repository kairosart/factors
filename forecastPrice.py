""" Implement price forecasting for a stock """

import os
import numpy as np
import pandas as pd
from flask import render_template
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree, metrics, neighbors
from sklearn.model_selection import cross_val_score
from math import sqrt
from indicators import plot_stock_prices_prediction, get_momentum, get_sma, get_RSI, plot_stock_prices, \
    plot_stock_prices_prediction_ARIMA
from util import fetchOnlineData, get_data, df_to_cvs, slice_df
import datetime as dt
from statsmodels.tsa.arima_model import ARIMAResults

def showforcastpricesvalues(request):

    # Get symbol
    symbol = str(request.get('ticker_select'))

    # Get Forecast date
    forecast_date = str(request.get('forecastDate'))
    forecast_date = dt.datetime.strptime(forecast_date, '%m/%d/%Y')

    # Get Forecast model
    forecast_model = str(request.get('model_Selection'))

    # Get Forecast time
    forecast_time = int(request.get('forecast_Time'))

    # Get Lookback
    forecast_lookback = int(request.get('look_Back'))


    # Get 1 year of data to train and test
    start_d = forecast_date - dt.timedelta(forecast_lookback)
    yesterday = dt.date.today() - dt.timedelta(1)

    try:
        download = fetchOnlineData(start_d, symbol, yesterday)
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


    # Get indicators
    sym_mom, sma, q, rsi_value = get_indicators(normed, symbol)


    # Create momentum column
    normed['Momentum'] = sym_mom

    # Create SMA column
    normed['SMA'] = sma

    # Create SMA column
    normed['RSI'] = rsi_value

    # Clean nan values
    normed = normed.fillna(0)

    # Sort dataframe by index
    normed.sort_index()

    corr_df = normed.corr(method='pearson')
    print("--------------- CORRELATIONS ---------------")
    print(corr_df)

    # Define X and y
    feature_cols = ['Momentum', 'RSI']
    X = normed[feature_cols]
    y = normed[symbol]

    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

    '''
    # Use only to get the best parameter for max_depth
    # Loop through a few different max depths and check the performance
    for d in [3, 5, 10]:
        # Create the tree and fit it
        decision_tree = tree.DecisionTreeRegressor(max_depth=d)
        decision_tree.fit(X_train, y_train)

        # Print out the scores on train and test
        print('max_depth=', str(d))
        print(decision_tree.score(X_train, y_train))
        print(decision_tree.score(X_test, y_test), '\n')
    '''

    def model_fit_pred(X_train, y_train, X_test):
        '''Fit a model and get predictions and metrics'''

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Measuring predictions

        # Accuracy
        scores = cross_val_score(model, X_test, y_test, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

        # Coefficient of determination R^2
        '''
        The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
        '''
        coef_deter = model.score(X_train, y_train)
        print('Coefficient of determination R^2: %s' % coef_deter)

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
        mae = metrics.mean_absolute_error(y_test, y_pred)
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

        # Mean squared error
        '''
        A mean squared error of zero indicates perfect skill, or no error.
        '''
        mse = metrics.mean_squared_error(y_test, y_pred)
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

        # Root mean squared error
        '''
        As with the mean squared error, an RMSE of zero indicates no error.
        '''
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        results = pd.DataFrame({'Price': y_test, 'Price prediction': y_pred})
        results.sort_index(inplace=True)

        return results, coef_deter, forecast_errors, bias, mae, mse, rmse

    # Decision Tree Regressor
    if forecast_model == '1':
        model = tree.DecisionTreeRegressor(max_depth=10)
        results, coef_deter, forecast_errors, bias, mae, mse, rmse = model_fit_pred(X_train, y_train, X_test)

    # KNN
    elif forecast_model == '2':
        params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
        knn = neighbors.KNeighborsRegressor()
        print('KNN: %s' % knn)
        model = GridSearchCV(knn, params, cv=5)
        results, coef_deter, forecast_errors, bias, mae, mse, rmse = model_fit_pred(X_train, y_train, X_test)

    # ARIMA
    elif forecast_model == '3':
        # load model
        loaded = ARIMAResults.load('arima_model.pkl')
        forecast = loaded.forecast(steps=forecast_time)[0]

        # Setting prediction dataframe
        dates = pd.date_range(forecast_date, periods=forecast_time)
        df = pd.DataFrame(forecast)
        df['Dates'] = dates
        df.set_index('Dates', inplace=True)
        df.rename(columns={0: 'Price'}, inplace=True)

        # Setting lookback dataframe
        dates = pd.date_range(forecast_lookback, forecast_date)

        # Create a dataframe with adjusted close prices for the symbol and for cash
        df_prices = slice_df(portf_value, dates)
        df_prices

    # TODO Calculate forecast after today
    '''
    # Create forecasting for future prices
    for i in range(forecast_time):
        # Calculate next price
        # TODO Only get the first price, the following are all the same
        # Data to predict the next price
        X_test = np.array([sym_mom[-1], rsi_value[-1]])
        next_price = model.predict(X_test.reshape(1, -1))

        # Add data to normed dataframe
        last_date = normed.iloc[[-1]].index
        t = last_date + dt.timedelta(days=1)
        next_day = pd.to_datetime(t, unit='s')

        # Get indicators
        sym_mom, sma, q, rsi_value = get_indicators(normed, symbol)

        # Create a date column to reindex
        normed['date'] = normed.index
        normed.loc[len(normed.index)] = [float(next_price), "NaN", "NaN", "NaN", next_day[0]]

        # Update normed
        normed.at[normed.index[-1], 'Momentum'] = float(sym_mom[-1])
        normed.at[normed.index[-1], 'SMA'] = sma[-1]
        normed.at[normed.index[-1], 'RSI'] = float(rsi_value[-1])

        # Reindex
        normed.set_index('date', inplace=True)
    '''




    if forecast_model == '3':
        # TODO ARIMA CHART

        plot_prices_pred = plot_stock_prices_prediction_ARIMA(df.index, df['Price'], symbol)
        return symbol, start_d, yesterday, plot_prices_pred
    else:
        # Plot prediction
        plot_prices_pred = plot_stock_prices_prediction(results.index, results['Price'], results['Price prediction'])
        return symbol, start_d, yesterday, plot_prices_pred, coef_deter, forecast_errors, bias, mae, mse, rmse


def get_indicators(normed, symbol):

    # Compute momentum
    sym_mom = get_momentum(normed[symbol], window=10)

    # ****Relative Strength Index (RSI)****
    # Compute RSI
    rsi_value = get_RSI(normed[symbol], 7)

    # ****Simple moving average (SMA)****
    # Compute SMA
    sma, q = get_sma(normed[symbol], window=10)
    return sym_mom, sma, q, rsi_value


