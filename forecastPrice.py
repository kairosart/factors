""" Implement price forecasting for a stock """
import pickle

import numpy as np
import pandas as pd
import datetime as dt

from keras.models import load_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree, metrics, neighbors
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from indicators import plot_stock_prices_prediction, get_indicators, \
    plot_stock_prices_prediction_ARIMA, plot_stock_prices_prediction_LSTM, get_momentum
from statsmodels.tsa.arima_model import ARIMAResults

from util import slice_df, create_dataset


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

    # Decision Tree (XGBoost)
    if forecast_model == '1':
        # load model
        model = pickle.load(open("./xgboost.pkl", "rb"))

        # Normalize the prices Dataframe
        normed = portf_value.copy()
        # normed = scaling_data(normed, symbol)

        normed['date'] = portf_value.index
        #normed.set_index('date', inplace=True)
        normed.rename(columns={'Adj Close': symbol}, inplace=True)

        # Clean nan values
        normed = normed.fillna(0)

        # Sort dataframe by index
        normed.sort_index()

        # Bussines days
        start = forecast_date.strftime("%Y-%m-%d")
        rng = pd.date_range(pd.Timestamp(start), periods=forecast_time, freq='B')
        bussines_days = rng.strftime('%Y-%m-%d %H:%M:%S')

        # Get indicators
        sym_mom, sma, q, rsi_value = get_indicators(normed, symbol)

        dataset = pd.DataFrame()

        # Create momentum column
        dataset['Momentum'] = sym_mom.values;

        # Create RSI column
        dataset['RSI'] = rsi_value;

        # Clean nan values
        dataset = dataset.fillna(0)

        for i in bussines_days:
            # Get indicators
            sym_mom, sma, q, rsi_value = get_indicators(normed, symbol)

            # Calculate price
            prediction = model.predict(dataset)

            # Add last value of result to normed
            p = prediction[-1]
            print('Prediction: ', p)
            normed.loc[len(normed)] = [p, i]

            # Add last indicators to dataset
            dataset.loc[len(dataset)] = [sym_mom.values, rsi_value]

        normed.set_index('date', inplace=True);

        # Plot chart
        plot_prices_pred = plot_stock_prices_prediction_LSTM(df_prices, normed, symbol)
        return symbol, start_d, forecast_date, plot_prices_pred

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

        # Plot chart
        plot_prices_pred = plot_stock_prices_prediction_ARIMA(df_prices, df, symbol)
        return symbol, start_d, forecast_date, plot_prices_pred, model_sumary


    # LSTM
    if forecast_model == '4':
        # load_model
        model = load_model('./lstm_model')

        # Lookback data
        lookback_date = dt.date.today() - dt.timedelta(forecast_lookback)
        dates = pd.date_range(lookback_date, periods=forecast_lookback)
        df_prices = slice_df(portf_value, dates)

        # Bussines days
        start = forecast_date.strftime("%Y-%m-%d")
        rng = pd.date_range(pd.Timestamp(start),  periods=forecast_time, freq='B')
        bussines_days = rng.strftime('%Y-%m-%d %H:%M:%S')

        # Setting prediction dataframe cols and list for adding rows to dataframe
        cols = ['Price', 'date']
        lst = []

        # Create date column to save next date
        portf_value['date'] = portf_value.index

        for i in bussines_days:
            # load the dataset
            dataset = np.array(portf_value.iloc[:, 0].tolist())[np.newaxis]
            dataset = dataset.T
            dataset = dataset.astype('float32')

            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)

            # prepare the X and Y label
            X, y = create_dataset(dataset)
            # Take 80% of data as the training sample and 20% as testing sample
            trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.20, shuffle=False)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

            # Prediction
            testPredict = model.predict(testX)
            futurePredict = model.predict(np.asarray([[testPredict[-1]]]))
            futurePredict = scaler.inverse_transform(futurePredict)
            prediction = futurePredict.item(0)


            # Adding last prediction to portf_value
            portf_value.loc[len(portf_value)] = [prediction, i]

            # Adding value to predictions dataframe
            lst.append([prediction, i])
            df = pd.DataFrame(lst, columns=cols)

        df.set_index('date', inplace=True)


        # Plot chart
        plot_prices_pred = plot_stock_prices_prediction_LSTM(df_prices, df, symbol)
        return symbol, start_d, forecast_date, plot_prices_pred

        #TODO Add results to chart page


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
    #TODO Change the way of predicting. Instead of X_test use forecast_time
    def model_fit_pred(X_train, y_train, X_test):
        '''Fit a model and get predictions and metrics'''

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Measuring predictions

        # Accuracy
        scores = cross_val_score(model, X_test, y_test, cv=7)
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





