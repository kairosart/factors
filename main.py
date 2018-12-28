import math
import os
import numpy as np

from flask import Flask, render_template, session, jsonify, request, flash

from sklearn.metrics import accuracy_score, explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from forecastPrice import showforcastpricesvalues
from form import StartValuesForm, get_tickers, pricesForecast
import pandas as pd
import datetime as dt
from sklearn import datasets, svm, metrics
from markupsafe import Markup
import fix_yahoo_finance as yf


yf.pdr_override()

from util import create_df_benchmark, fetchOnlineData, get_data, \
    scaling_data, symbol_to_path
from strategyLearner import strategyLearner
from marketsim import compute_portvals_single_symbol, market_simulator
from indicators import get_momentum, get_sma, get_sma_indicator, get_rolling_mean, get_rolling_std, get_bollinger_bands, \
    compute_bollinger_value, get_RSI, plot_cum_return, plot_momentum, plot_sma_indicator, plot_rsi_indicator, \
    plot_momentum_sma_indicator, plot_stock_prices, plot_bollinger, plot_norm_data_vertical_lines, \
    plot_stock_prices_prediction

# prep
from sklearn.model_selection import train_test_split
from sklearn import tree


app = Flask(__name__)
app.config.from_object('config')

@app.route('/')
def home():
    return render_template(
    # name of template
    "index.html"
    )

@app.route('/overview')
def overview():
    return render_template(
    # name of template
    "overview.html"
    )



# Show indicators values
@app.route('/showvalues', methods=['POST', 'GET'])
def showvalues():

    # Get portfolio values from Yahoo
    symbol = request.form.get('ticker_select', type=str)

    # Get 1 year of data to train and test
    start_d = dt.date.today() - dt.timedelta(365)
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
                        "startValuesForm.html",
                        error=True)
            except:
                return render_template(
                    # name of template
                    "startValuesForm.html",
                    error=True)

    portf_value = get_data(symbol)


    # ****Stock prices chart****
    plot_prices = plot_stock_prices(portf_value.index, portf_value[symbol], symbol)


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

    # Session variables
    session['start_val'] = request.form['start_val']
    session['symbol'] = symbol
    session['start_d'] = start_d.strftime('%Y/%m/%d')
    session['num_shares'] = request.form['num_shares']
    session['commission'] = request.form['commission']
    session['impact'] = request.form['impact']
    session['type'] = request.form['forecast']

    if session['type'] == "showvalues":
        # Price movements

        # Create momentum chart
        plot_mom = plot_momentum(portf_value.index, normed[symbol], symbol, sym_mom, "Momentum Indicator", (12, 8))


        # Plot raw symbol values, rolling mean and Bollinger Bands
        dates = pd.date_range(start_d, yesterday)
        plot_boll = plot_bollinger(dates, portf_value.index, portf_value[symbol], symbol, upper_band, lower_band,
                                   rm_JPM,
                                   num_std=1, title="Bollinger Indicator", fig_size=(12, 6))

        # Plot symbol values, SMA and SMA quality
        plot_sma = plot_sma_indicator(dates, portf_value.index, normed[symbol], symbol, sma, q,
                                      "Simple Moving Average (SMA)")


        # Plot RSI
        plot_rsi = plot_rsi_indicator(dates, portf_value.index, portf_value[symbol], symbol, rsi_value, window=14,
                                      title="RSI Indicator", fig_size=(12, 6))
        return render_template(
        # name of template
        "stockpriceschart.html",

        # now we pass in our variables into the template
        start_val = request.form['start_val'],
        symbol = symbol,
        commission = request.form['commission'],
        impact = request.form['impact'],
        num_shares = request.form['num_shares'],
        start_date = start_d,
        end_date = yesterday,
        tables=[portf_value.to_html()],
        titles = ['na', 'Stock Prices '],
        div_placeholder_stock_prices = Markup(plot_prices),
        div_placeholder_momentum = Markup(plot_mom),
        div_placeholder_bollinger = Markup(plot_boll),
        div_placeholder_sma = Markup(plot_sma),
        div_placeholder_rsi = Markup(plot_rsi)
        )
    else:
        #TODO Delete this part

        # Price forecasting

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
        return render_template(
            # name of template
            "forecastPrices.html",
            # now we pass in our variables into the template
            start_val=request.form['start_val'],
            symbol=symbol,
            commission=request.form['commission'],
            impact=request.form['impact'],
            num_shares=request.form['num_shares'],
            start_date=start_d,
            end_date=yesterday,
            div_placeholder_stock_prices_pred=Markup(plot_prices_pred),
            titles=['na', 'Stock Prices '],
        )


# Initial form to get indicators values
@app.route('/form', methods = ['GET', 'POST'])
def introStartValues():
    form = StartValuesForm()

    # Get ticker name list
    tickers = get_tickers('nasdaq_tickers_name')

    if request.method == 'POST':
        if form.validate() == False:
            flash('All fields are required.')
            return render_template('startValuesForm.html', form=form)
        else:
            return render_template('success.html')
    elif request.method == 'GET':

        return render_template('startValuesForm.html',
                               form = form,
                               data=tickers)

# Initial form to get values for price forecasting
@app.route('/showforecastform', methods = ['GET', 'POST'])
def showforecastform():
    form = pricesForecast()

    # Get ticker name list
    tickers = get_tickers('nasdaq_tickers_name')

    if request.method == 'POST':
        result = request.form
        symbol, start_d, yesterday, plot_prices_pred = showforcastpricesvalues(result)
        return render_template(
            # name of template
            "forecastPrices.html",
            # now we pass in our variables into the template
            symbol=symbol,
            start_date=start_d,
            end_date=yesterday,
            div_placeholder_stock_prices_pred=Markup(plot_prices_pred),
            titles=['na', 'Stock Prices '],
        )
    elif request.method == 'GET':

        return render_template('pricesForecastForm.html',
                               form=form,
                               data=tickers)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
