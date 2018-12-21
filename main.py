import math
import os
import numpy as np

from flask import Flask, render_template, session, jsonify, request, flash
from sklearn.metrics import accuracy_score, explained_variance_score
from sklearn.preprocessing import StandardScaler

from form import StartValuesForm
import pandas as pd
import datetime as dt
from sklearn import datasets, svm
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

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

# Load Dataset from scikit-learn.
digits = datasets.load_digits()

# Show values
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



        model = LinearRegression()
        model.fit(X_train, y_train)

        # We then use the model to make predictions based on the test values of x
        y_pred = model.predict(X_test)

        # Now that we have trained the model, we can print the coefficient of x that it has predicted
        print('Coefficient: \n', model.coef_)

        # calculate accuracy
        print('Accuracy:', explained_variance_score(y_test, y_pred, multioutput='uniform_average'))
        # Now, we can calculate the models accuracy metrics based on what the actual value of y was
        print("Mean squared error: %.2f"
              % mean_squared_error(y_pred, y_test))
        print('r_2 statistic: %.2f' % r2_score(y_pred, y_test))

        # Predictions
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

# Training
@app.route('/benchmark', methods=['POST', 'GET'])
def training():
    # **** Training ***
    # Getting session variables
    start_val = int(session.get('start_val', None))
    symbol = session.get('symbol', None)
    num_shares = int(session.get('num_shares', None))
    commission = float(session.get('commission', None))
    impact = float(session.get('impact', None))


    # Create a dataframe from csv file
    df = get_data(symbol)


    # We'll get 80% data to train
    split_percentage = 0.8
    split = int(split_percentage * len(df))

    # Train data set
    df_training = df[:split]

    # Test data set
    df_testing = df[split:]

    # Training dates
    start_date_training = df_training.index[0]
    end_date_training = df_training.index[-1]

    # Testing dates
    start_date_testing = df_testing.index[0]
    end_date_testing = df_testing.index[-1]

    # Get a dataframe of benchmark data. Benchmark is a portfolio starting with
    # $100,000, investing in 1000 shares of symbol and holding that position
    df_benchmark_trades = create_df_benchmark(df_training, num_shares)

    # **** Training ****
    # Train a StrategyLearner
    stl = strategyLearner(num_shares=num_shares, impact=impact,
                          commission=commission, verbose=True,
                          num_states=3000, num_actions=3)

    epochs, cum_returns = stl.add_evidence(df_training, symbol, start_val=start_val, start_date=start_date_training, end_date=end_date_training)
    df_trades = stl.test_policy(df_training, symbol, start_date=start_date_training,
                                end_date=end_date_training)

    plot_cum = plot_cum_return(epochs, cum_returns)

    # Retrieve performance stats via a market simulator
    print("Performances during training period for {}".format(symbol))
    print("Date Range: {} to {}".format(start_date_training, end_date_training))
    orders_count, sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, final_value, cum_ret_bm, avg_daily_ret_bm, std_daily_ret_bm, sharpe_ratio_bm, final_value_bm, portvals, portvals_bm, df_orders = \
        market_simulator(df_training, df_trades, df_benchmark_trades, symbol=symbol,
                     start_val=start_val, commission=commission, impact=impact)



    plot_norm_data = plot_norm_data_vertical_lines(
                            df_orders,
                            portvals,
                            portvals_bm,
                            vert_lines=False,
                            title="Training Portfolio Value",
                            xtitle="Dates",
                            ytitle="Value ($)")


    # **** Testing ****
    # Out-of-sample or testing period: Perform similiar steps as above,
    # except that we don't train the data (i.e. run add_evidence again)

    df_benchmark_trades = create_df_benchmark(df_testing, num_shares)
    df_trades = stl.test_policy(df_testing, symbol,  start_date=start_date_testing,
                                end_date=end_date_testing)
    print("\nPerformances during testing period for {}".format(symbol))
    print("Date Range: {} to {}".format(start_date_testing, end_date_testing))


    # Retrieve performance stats via a market simulator
    test_orders_count, test_sharpe_ratio, test_cum_ret, test_std_daily_ret, test_avg_daily_ret, test_final_value, test_cum_ret_bm, test_avg_daily_ret_bm, test_std_daily_ret_bm, test_sharpe_ratio_bm, test_final_value_bm, test_portvals, test_portvals_bm, test_df_orders = \
        market_simulator(df_testing, df_trades, df_benchmark_trades, symbol=symbol,
                     start_val=start_val, commission=commission, impact=impact)

    plot_norm_data_test = plot_norm_data_vertical_lines(
                            test_df_orders,
                            test_portvals,
                            test_portvals_bm,
                            vert_lines=True,
                            title="Testing Portfolio Value",
                            xtitle="Dates",
                            ytitle="Value ($)")

    return render_template(
        # name of template
        "training.html",

        # Training data
        start_date = start_date_training,
        end_date = end_date_training,
        symbol = symbol,
        div_placeholder_plot_cum = Markup(plot_cum),
        div_placeholder_plot_norm_data = Markup(plot_norm_data),
        orders_count_p = orders_count,
        sharpe_ratio_p = round(sharpe_ratio, 3),
        cum_ret_p = round(cum_ret, 3),
        std_daily_ret_p = round(std_daily_ret, 3),
        avg_daily_ret_p = round(avg_daily_ret, 3),
        final_value_p = round(final_value, 3),
        sharpe_ratio_b = round(sharpe_ratio_bm, 3),
        cum_ret_b = round(cum_ret_bm, 3),
        std_daily_ret_b = round(std_daily_ret_bm, 3),
        avg_daily_ret_b = round(avg_daily_ret_bm, 3),
        final_value_b = round(final_value_bm, 3),

        # Testing datasets
        start_date_testing = start_date_testing,
        end_date_testing = end_date_testing,
        div_placeholder_plot_norm_data_testing = Markup(plot_norm_data_test),
        orders_count_p_testing = test_orders_count,
        sharpe_ratio_p_testing = round(test_sharpe_ratio, 3),
        cum_ret_p_testing = round(test_cum_ret, 3),
        std_daily_ret_p_testing = round(test_std_daily_ret, 3),
        avg_daily_ret_p_testing = round(test_avg_daily_ret, 3),
        final_value_p_testing = round(test_final_value, 3),
        sharpe_ratio_b_testing = round(test_sharpe_ratio_bm, 3),
        cum_ret_b_testing = round(test_cum_ret_bm, 3),
        std_daily_ret_b_testing = round(test_std_daily_ret_bm, 3),
        avg_daily_ret_b_testing = round(test_avg_daily_ret_bm, 3),
        final_value_b_testing = round(test_final_value_bm, 3)

    )


# Initial form to get values
@app.route('/form', methods = ['GET', 'POST'])
def introStartValues():
    form = StartValuesForm()

    # Get ticker name list
    tickers = form.get_tickers('nasdaq_tickers_name')

    if request.method == 'POST':
        if form.validate() == False:
            flash('All fields are required.')
            return render_template('startValuesForm.html', form = form)
        else:
            return render_template('success.html')
    elif request.method == 'GET':

        return render_template('startValuesForm.html',
                               form = form,
                               data=tickers)



if __name__ == '__main__':
    app.run(host='0.0.0.0')
