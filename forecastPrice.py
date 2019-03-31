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
from sklearn import preprocessing


from indicators import plot_stock_prices_prediction, get_indicators, \
    plot_stock_prices_prediction_ARIMA, plot_stock_prices_prediction_LSTM, get_momentum, \
    plot_stock_prices_prediction_XGBoost
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults

from util import slice_df, create_dataset

# TA Library (https://github.com/bukosabino/ta)
from ta import *

def prepare_data_for_metrics(portf_value, symbol):
    """

    :param portf_value: Dataframe with prices
    :param symbol: Stock symbol
    :return: Splitting training and testing sets
    """
    # Normalize the prices Dataframe
    normed = portf_value.copy()
    # normed = scaling_data(normed, symbol)

    normed['date'] = portf_value.index
    normed.set_index('date', inplace=True)
    normed.rename(columns={'Adj Close': symbol}, inplace=True)

    # Get indicators
    sym_mom, sma, q, rsi_value = get_indicators(normed, symbol)

    # Create momentum column
    normed['Momentum'] = sym_mom

    # Create SMA column
    normed['RSI'] = rsi_value

    # Clean nan values
    normed = normed.fillna(0)

    # Sort dataframe by index
    normed.sort_index()

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(normed)

    # Create dataset dataframe
    df_normed = pd.DataFrame(dataset, index=range(dataset.shape[0]),
                      columns=range(dataset.shape[1]))

    # Rename columns
    df_normed.rename(columns={0: symbol}, inplace=True)
    df_normed.rename(columns={1: 'Momentum'}, inplace=True)
    df_normed.rename(columns={2: 'RSI'}, inplace=True)


    # Define X and y
    feature_cols = ['Momentum', 'RSI']
    X = df_normed[feature_cols]
    y = df_normed[symbol]

    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

    return X_train, X_test, y_train, y_test

# TODO Change the way of predicting. Instead of X_test use forecast_time
def model_fit_pred(model, X_train, y_train, X_test, y_test):
    '''Fit a model and get predictions and metrics'''

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Measuring predictions

    # Accuracy
    scores = cross_val_score(model, X_test, y_test)
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

def showforcastpricesvalues(symbol, portf_value, forecast_model, forecast_time, start_d, forecast_date, forecast_lookback):
    '''

    :param symbol: Stock symbol
    :param portf_value: Prices dataframe
    :param forecast_model: Model for forecasting
    :param forecast_time: Number of days to forecast in the future
    :param start_d: Lookback date
    :param forecast_date: Forecasting date
    :param forecast_lookback: Number of days to lookback
    :return: Prices Plot.
    '''

    # XGBoost (1 day forecasting)
    if forecast_model == 'model1':
        ## Indicators to use
        '''
            * others_cr: Cumulative Return. (Close)
            * trend_ema_fast: Fast Exponential Moving Averages(EMA) (Close)
            * volatility_kcl: Keltner channel (KC) (High, Low, Close)'''

        # load model
        model = pickle.load(open("./xgboost.pkl", "rb"))

        # Lookback data
        lookback_date = dt.date.today() - dt.timedelta(forecast_lookback)
        dates = pd.date_range(lookback_date, periods=forecast_lookback)
        #df_prices = slice_df(portf_value, dates)
        df_prices = portf_value[['Adj Close']].copy()

        # Create datafrane with all TA indicators
        df = add_all_ta_features(portf_value, "Open", "High", "Low", "Close", "Volume", fillna=True)

        # Delete unuseful columns
        del df['Open']
        del df['High']
        del df['Low']
        del df['Close']
        del df['Volume']

        # Create 'date' column for posterior index
        df['date'] = df.index

        # Rename column for correlation matrix. Can't have spaces.
        df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)

        # Reset index
        df.reset_index(inplace=True)

        # Scale data for using reg:logistic as array
        scaler = MinMaxScaler(feature_range=(0, 1))
        features = df[['others_cr', 'trend_ema_fast', 'volatility_kcl']]
        dataset_scaled = scaler.fit_transform(features)

        # Scale X_test (Adj_Close)
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        feature = df[['Adj_Close']]
        X_test_scaled = scaler1.fit_transform(feature)

        # Next Bussines days
        start = forecast_date.strftime("%Y-%m-%d")
        rng = pd.date_range(pd.Timestamp(start), periods=1, freq='B')
        bussines_days = rng.strftime('%Y-%m-%d')

        # Setting prediction dataframe cols and list for saving predictions
        cols = ['Price', 'date']
        lst = []

        # Calculate price
        prediction = model.predict(dataset_scaled)
        preds = scaler1.inverse_transform(prediction.reshape(-1, 1))

        # Convert array to series
        mylist = preds.tolist()
        p = mylist[-1][-1]

        # Adding value to predictions dataframe for plotting
        lst.append([p, bussines_days.values[0]])
        df_predictions = pd.DataFrame(lst, columns=cols)

        # TODO Calculate metrics

        # Daily return
        # Get last row of df_prices
        df_predictions.index  = df_predictions['date']
        last_date = df_prices.loc[df_prices.index[-1]].name
        last_date = last_date.strftime("%Y-%m-%d")
        last_price = df_prices.loc[df_prices.index[-1]][0]

        # Add last date and price to dataframe
        df_predictions.loc[len(df_predictions)] = [last_price, last_date ]
        df_predictions.set_index('date', inplace=True)
        df_predictions.index = pd.to_datetime(df_predictions.index, format='%Y-%m-%d')
        df_predictions.index = df_predictions.index.strftime("%Y-%m-%d")
        df_predictions.sort_index(inplace=True)


        # Daily Return Percentage change between the current and a prior element.
        drp = df_predictions.pct_change(1)
        # Rename price column to % variation
        drp.rename(columns={'Price': '%\u25B3'}, inplace=True)

        # Compute the price difference of two elements
        diff = df_predictions.diff()
        # Rename price column to $ variation
        diff.rename(columns={'Price': '$\u25B3'}, inplace=True)

        # Concat diff with drp
        metric = pd.concat([diff, drp], axis=1)

        # Concat forecast prices with metric
        metric = pd.concat([df_predictions, diff, drp], axis=1)
        metric.rename(columns={'Price': 'Forecast'}, inplace=True)

        # Clean NaN
        metric = metric.fillna(0)

        # Set decimals to 2
        metric['Forecast'] = metric['Forecast'].apply(lambda x: round(x, 2))
        metric['%\u25B3'] = metric['%\u25B3'].apply(lambda x: round(x, 2))
        metric['$\u25B3'] = metric['$\u25B3'].apply(lambda x: round(x, 2))

        # Plot chart
        plot_prices_pred = plot_stock_prices_prediction_XGBoost(df_prices, df_predictions, symbol)
        return symbol, start_d, forecast_date, plot_prices_pred, metric

    # KNN Model (1 day forecasting)
    if forecast_model == 'model2':
        ## Indicators to use
        '''
            * others_cr: Cumulative Return. (Close)
            * trend_ema_fast: Fast Exponential Moving Averages(EMA) (Close)
            * volatility_kcl: Keltner channel (KC) (High, Low, Close)'''

        # load model
        model = pickle.load(open("./knn.pkl", "rb"))

        # Lookback data
        lookback_date = dt.date.today() - dt.timedelta(forecast_lookback)
        dates = pd.date_range(lookback_date, periods=forecast_lookback)
        # df_prices = slice_df(portf_value, dates)
        df_prices = portf_value[['Adj Close']].copy()

        # Create datafrane with all TA indicators
        df = add_all_ta_features(portf_value, "Open", "High", "Low", "Close", "Volume", fillna=True)

        # Delete unuseful columns
        del df['Open']
        del df['High']
        del df['Low']
        del df['Close']
        del df['Volume']

        # Create 'date' column for posterior index
        df['date'] = df.index

        # Rename column for correlation matrix. Can't have spaces.
        df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)

        # Reset index
        df.reset_index(inplace=True)

        # Scale data for using reg:logistic as array
        scaler = MinMaxScaler(feature_range=(0, 1))
        features = df[['others_cr', 'trend_ema_fast', 'volatility_kcl']]
        dataset_scaled = scaler.fit_transform(features)

        # Scale X_test (Adj_Close)
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        feature = df[['Adj_Close']]
        X_test_scaled = scaler1.fit_transform(feature)

        # Next Bussines days
        start = forecast_date.strftime("%Y-%m-%d")
        rng = pd.date_range(pd.Timestamp(start), periods=1, freq='B')
        bussines_days = rng.strftime('%Y-%m-%d')

        # Setting prediction dataframe cols and list for saving predictions
        cols = ['Price', 'date']
        lst = []

        # Calculate price
        prediction = model.predict(dataset_scaled)
        preds = scaler1.inverse_transform(prediction.reshape(-1, 1))

        # Convert array to series
        mylist = preds.tolist()
        p = mylist[-1][-1]

        # Adding value to predictions dataframe for plotting
        lst.append([p, bussines_days.values[0]])
        df_predictions = pd.DataFrame(lst, columns=cols)

        # Daily return
        # Get last row of df_prices
        df_predictions.index = df_predictions['date']
        last_date = df_prices.loc[df_prices.index[-1]].name
        last_date = last_date.strftime("%Y-%m-%d")
        last_price = df_prices.loc[df_prices.index[-1]][0]

        # Add last date and price to dataframe
        df_predictions.loc[len(df_predictions)] = [last_price, last_date]
        df_predictions.set_index('date', inplace=True)
        df_predictions.index = pd.to_datetime(df_predictions.index, format='%Y-%m-%d')
        df_predictions.index = df_predictions.index.strftime("%Y-%m-%d")
        df_predictions.sort_index(inplace=True)

        # Daily Return Percentage change between the current and a prior element.
        drp = df_predictions.pct_change(1)
        # Rename price column to % variation
        drp.rename(columns={'Price': '%\u25B3'}, inplace=True)

        # Compute the price difference of two elements
        diff = df_predictions.diff()
        # Rename price column to $ variation
        diff.rename(columns={'Price': '$\u25B3'}, inplace=True)

        # Concat diff with drp
        metric = pd.concat([diff, drp], axis=1)

        # Concat forecast prices with metric
        metric = pd.concat([df_predictions, diff, drp], axis=1)
        metric.rename(columns={'Price': 'Forecast'}, inplace=True)

        # Clean NaN
        metric = metric.fillna(0)

        # Set decimals to 2
        metric['Forecast'] = metric['Forecast'].apply(lambda x: round(x, 2))
        metric['%\u25B3'] = metric['%\u25B3'].apply(lambda x: round(x, 2))
        metric['$\u25B3'] = metric['$\u25B3'].apply(lambda x: round(x, 2))

        # Plot chart
        plot_prices_pred = plot_stock_prices_prediction_XGBoost(df_prices, df_predictions, symbol)
        return symbol, start_d, forecast_date, plot_prices_pred, metric

    # ARIMA
    if forecast_model == 'model3':
        # Rolling forecasts
        '''                                                                 
        Load the model and use it in a rolling-forecast manner,             
        updating the transform and model for each time step.                
        This is the preferred method as it is how one would use             
        this model in practice as it would achieve the best performance.    
        '''

        # Setting dates and prices dataframe
        lookback_date = dt.date.today() - dt.timedelta(forecast_lookback)
        dates = pd.date_range(lookback_date, periods=forecast_lookback + 1)
        df_prices = slice_df(portf_value, dates)

        # Bussines days
        # Check whether today is on portf_value
        lvi =  pd.Timestamp.date(portf_value.last_valid_index())
        today = dt.date.today()
        if lvi == today:
            start = forecast_date + dt.timedelta(1)
        else:
            start = forecast_date
        rng = pd.date_range(pd.Timestamp(start),  periods=forecast_time, freq='B')
        bussines_days = rng.strftime('%Y-%m-%d')

        # Setting prediction dataframe cols and list for adding rows to dataframe
        cols = ['Price', 'date', 'lower_band', 'upper_band', 'Std. Error']
        lst = []

        # Create date column to save next date
        portf_value['date'] = portf_value.index

        # Forecasting for every business day
        for i in bussines_days:
            # load the dataset
            dataset = np.array(portf_value.iloc[:, 0].tolist())[np.newaxis]
            dataset = dataset.T
            dataset = dataset.astype('float32')

            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)

            # predict
            model = ARIMA(dataset, order=(4,0,1))
            model_fit = model.fit(disp=0)
            yhat, se, conf = model_fit.forecast(alpha=0.05)

            # Prediction Inverse scale
            prediction = yhat[0].reshape(-1, 1)
            futurePredict = scaler.inverse_transform(prediction)

            # Confidence intervals inverse transform
            inv_scaled_conf =  scaler.inverse_transform(conf[0][0].reshape(1, -1))
            inv_scaled_conf1 =  scaler.inverse_transform(conf[0][1].reshape(1, -1))

            # Confident intervals
            lower_band = inv_scaled_conf[0][0]
            upper_band = inv_scaled_conf1[0][0]

            # Adding last prediction to portf_value
            prediction = futurePredict.item(0)
            portf_value.loc[len(portf_value)] = [prediction, i]

            # Adding value to predictions dictionary
            lst.append([prediction, i, lower_band, upper_band, se])

            # Setting dataframe for predictions and confident intervals
            df = pd.DataFrame(lst, columns=cols)

        # Order confidence values for plotting
        lower_list = df['lower_band'].tolist()
        lower_list.sort(reverse=True)
        upper_list = df['upper_band'].tolist()
        upper_list.sort()

        # Adding confidence bands data to dataframe
        df['lower_band'] = lower_list
        df['upper_band'] = upper_list
        df.set_index('date', inplace=True)
        df.rename(columns = {0:'Price'}, inplace=True)

        # Create ARIMA Report
        df_predictions = df[['Price']].copy()
        df_predictions['date'] = df_predictions.index
        last_date = df_prices.loc[df_prices.index[-1]].name
        last_date = last_date.strftime("%Y-%m-%d")
        last_price = df_prices.loc[df_prices.index[-1]][0]

        # Add last date and price to dataframe
        df_predictions.loc[len(df_predictions)] = [last_price, last_date]
        df_predictions.set_index('date', inplace=True)
        df_predictions.index = pd.to_datetime(df_predictions.index, format='%Y-%m-%d')
        df_predictions.index = df_predictions.index.strftime("%Y-%m-%d")
        df_predictions.sort_index(inplace=True)

        # Daily Return Percentage change between the current and a prior element.
        drp = df_predictions.pct_change(1)
        # Rename price column to % variation
        drp.rename(columns={'Price': '%\u25B3'}, inplace=True)

        # Compute the price difference of two elements
        diff = df_predictions.diff()
        # Rename price column to $ variation
        diff.rename(columns={'Price': '$\u25B3'}, inplace=True)

        # Concat diff with drp
        #metric = pd.concat([diff, drp], axis=1)

        # Concat forecast prices with metric
        metric = pd.concat([df_predictions, diff, drp], axis=1)
        metric.rename(columns={'Price': 'Forecast'}, inplace=True)

        # Clean NaN
        metric = metric.fillna(0)

        # Set decimals to 2
        metric['Forecast'] = metric['Forecast'].apply(lambda x: round(x, 2))
        metric['%\u25B3'] = metric['%\u25B3'].apply(lambda x: round(x, 3))
        metric['$\u25B3'] = metric['$\u25B3'].apply(lambda x: round(x, 3))


        # TODO Accuracy metrics https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

        # Plot chart
        plot_prices_pred = plot_stock_prices_prediction_ARIMA(df_prices, df, symbol)
        return symbol, start_d, forecast_date, plot_prices_pred, metric

    # LSTM
    if forecast_model == 'model4':
        # load_model
        model = load_model('./lstm_model')

        # Setting dates and prices dataframe
        lookback_date = dt.date.today() - dt.timedelta(forecast_lookback)
        dates = pd.date_range(lookback_date, periods=forecast_lookback + 1)
        df_prices = slice_df(portf_value, dates)

        # Bussines days
        # Check whether today is on portf_value
        lvi =  pd.Timestamp.date(portf_value.last_valid_index())
        today = dt.date.today()
        if lvi == today:
            start = forecast_date + dt.timedelta(1)
        else:
            start = forecast_date
        rng = pd.date_range(pd.Timestamp(start),  periods=forecast_time, freq='B')
        bussines_days = rng.strftime('%Y-%m-%d')

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



















