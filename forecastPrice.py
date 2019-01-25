""" Implement price forecasting for a stock """

import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree, metrics, neighbors
from sklearn.model_selection import cross_val_score
from indicators import plot_stock_prices_prediction, get_momentum, get_sma, get_RSI, plot_stock_prices, \
    plot_stock_prices_prediction_ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

from util import slice_df


def showforcastpricesvalues(symbol, portf_value, forecast_model, forecast_time, start_d, forecast_date, forecast_lookback):
    '''

    :param symbol: Stock symbol
    :param portf_value: Prices dataframe
    :param forecast_model: Model for forecasting
    :param forecast_time: Number of days to forecast in the future
    :param start_d: Lookback date
    :param forecast_date: Forecasting date
    :param forecast_lookback: Number of day to lookback
    :return: Summary and scores.
    '''

    # ARIMA
    if forecast_model == '3':
        # load model
        model = ARIMAResults.load('arima_model.pkl')
        forecast = model.forecast(steps=forecast_time)[0]

        # Lookback data
        lookback_date = dt.date.today() - dt.timedelta(forecast_lookback)
        dates = pd.date_range(lookback_date, periods=forecast_lookback)
        df_prices = slice_df(portf_value, dates)

        # Setting prediction dataframe
        dates = pd.date_range(forecast_date, periods=forecast_time)
        df = pd.DataFrame(forecast)
        df['Dates'] = dates
        df.set_index('Dates', inplace=True)
        df.rename(columns={0: 'Price'}, inplace=True)


        # TODO Confidence interval arc chart
        # ARIMA Model Results
        model_sumary = model.summary()
        plot_prices_pred = plot_stock_prices_prediction_ARIMA(df_prices, df, symbol)
        return symbol, start_d, forecast_date, plot_prices_pred, model_sumary

    # TODO Implement LSTM Method

    # Normalize the prices Dataframe
    normed = portf_value.copy()
    #normed = scaling_data(normed, symbol)

    normed['date'] = portf_value.index
    normed.set_index('date', inplace=True)
    normed.rename(columns={'Adj Close': symbol}, inplace=True)


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
    # KNN
    elif forecast_model == '2':
        params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
        knn = neighbors.KNeighborsRegressor()
        print('KNN: %s' % knn)
        model = GridSearchCV(knn, params, cv=5)

    results, coef_deter, forecast_errors, bias, mae, mse, rmse = model_fit_pred(X_train, y_train, X_test)

    # Plot prediction
    plot_prices_pred = plot_stock_prices_prediction(results.index, results['Price'], results['Price prediction'])
    return symbol, start_d, forecast_date, plot_prices_pred, coef_deter, forecast_errors, bias, mae, mse, rmse


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


