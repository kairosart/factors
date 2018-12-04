from flask import Flask, render_template, session, jsonify, request, flash
from form import StartValuesForm
import pandas as pd
import numpy as np
import random
import datetime as dt
from sklearn import datasets, svm
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import json
from markupsafe import Markup
import pickle

# To fetch data
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()

from util import create_df_benchmark, fetchOnlineData, get_learner_data_file
from strategyLearner import strategyLearner
from marketsim import compute_portvals_single_symbol, market_simulator
from indicators import get_momentum, get_sma, get_sma_indicator, get_rolling_mean, get_rolling_std, get_bollinger_bands, compute_bollinger_value, get_RSI, plot_cum_return,  plot_momentum, plot_sma_indicator, plot_rsi_indicator, plot_momentum_sma_indicator, plot_stock_prices, plot_bollinger, plot_norm_data_vertical_lines




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
    # Specify the start and end dates for this period.
    start_d = dt.date(2008, 1, 1)
    #end_d = dt.datetime(2018, 10, 30)
    yesterday = dt.date.today() - dt.timedelta(1)

    # Get portfolio values from Yahoo
    symbol = request.form['symbol']
    portf_value = fetchOnlineData(start_d, yesterday, symbol)


    # ****Stock prices chart****
    plot_prices = plot_stock_prices(portf_value.index, portf_value['Adj Close'], symbol)

    # ****Momentum chart****
    # Normalize the prices Dataframe
    normed = pd.DataFrame()
    normed['Adj Close'] = portf_value['Adj Close'].values / portf_value['Adj Close'].iloc[0];

    # Compute momentum
    sym_mom = get_momentum(normed['Adj Close'], window=10)

    # Create momentum chart
    plot_mom = plot_momentum(portf_value.index, normed['Adj Close'], sym_mom, "Momentum Indicator", (12, 8))

    # ****Bollinger Bands****
    # Compute rolling mean
    rm_JPM = get_rolling_mean(portf_value['Adj Close'], window=10)

    # Compute rolling standard deviation
    rstd_JPM = get_rolling_std(portf_value['Adj Close'], window=10)

    # Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm_JPM, rstd_JPM)

    # Plot raw symbol values, rolling mean and Bollinger Bands
    dates = pd.date_range(start_d, yesterday)
    plot_boll = plot_bollinger(dates, portf_value.index, portf_value['Adj Close'], symbol, upper_band, lower_band, rm_JPM,
                   num_std=1, title="Bollinger Indicator", fig_size=(12, 6))


    # ****Simple moving average (SMA)****
    # Compute SMA
    sma_JPM, q = get_sma(normed['Adj Close'], window=10)

    # Plot symbol values, SMA and SMA quality
    plot_sma = plot_sma_indicator(dates, portf_value.index, normed['Adj Close'], symbol, sma_JPM, q, "Simple Moving Average (SMA)")


    # ****Relative Strength Index (RSI)****
    # Compute RSI
    rsi_value = get_RSI(portf_value['Adj Close'])

    # Plot RSI
    plot_rsi =  plot_rsi_indicator(dates, portf_value.index, portf_value['Adj Close'], symbol, rsi_value, window=14,
                       title="RSI Indicator", fig_size=(12, 6))

    # Session variables
    session['start_val'] = request.form['start_val']
    session['symbol'] = request.form['symbol']
    session['start_d'] = start_d.strftime('%Y/%m/%d')
    session['num_shares'] = request.form['num_shares']
    session['commission'] = request.form['commission']
    session['impact'] = request.form['impact']


    return render_template(
    # name of template
	"stockpriceschart.html",

    # now we pass in our variables into the template
    start_val = request.form['start_val'],
    symbol = request.form['symbol'],
    commission = request.form['commission'],
    impact = request.form['impact'],
    num_shares = request.form['num_shares'],
    start_date = start_d,
    end_date = yesterday,
    tables=[portf_value.to_html(classes=symbol)],
    titles = ['na', 'Stock Prices '],
    div_placeholder_stock_prices = Markup(plot_prices),
    div_placeholder_momentum = Markup(plot_mom),
    div_placeholder_bollinger = Markup(plot_boll),
    div_placeholder_sma = Markup(plot_sma),
    div_placeholder_rsi = Markup(plot_rsi)

    )

# Training
@app.route('/benchmark', methods=['POST', 'GET'])
def training():
    # **** Training ***
    # Getting session variables
    start_val = session.get('start_val', None)
    symbol = session.get('symbol', None)
    start_d = session.get('start_d', None)
    start_d = dt.datetime.strptime(start_d, '%Y/%m/%d').date()
    #start_d = start_d.strftime("%Y/%m/%d")
    num_shares = session.get('num_shares', None)
    commission = session.get('commission', None)
    impact = session.get('impact', None)

    # Specify the start and end dates for this period. For traininig we'll get 80% of dates.
    n_days_training = ((dt.date.today()-start_d).days) / 3
    end_d = dt.date.today() - dt.timedelta(n_days_training)


    # Get benchmark data
    benchmark_prices = fetchOnlineData(start_d, end_d, symbol)

    # Create benchmark data: Benchmark is a portfolio starting with $100,000, investing in 1000 shares of symbol and holding that position
    df_benchmark_trades = create_df_benchmark(symbol, start_d, end_d, num_shares)
    print(df_benchmark_trades)
    
    # Train a StrategyLearner
    # Set verbose to True will print out and plot the cumulative return for each training epoch
    stl = strategyLearner(num_shares=num_shares, impact=impact,
                          commission=commission, verbose=True,
                          num_states=3000, num_actions=3)

    epochs, cum_returns = stl.add_evidence(df_prices=benchmark_prices, symbol=symbol, start_val=start_val)


    plot_cum = plot_cum_return(epochs, cum_returns)
    df_trades = stl.test_policy(symbol=symbol, start_date=start_d, end_date=end_d)
    print(df_trades)
    # Retrieve performance stats via a market simulator
    orders_count, sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, final_value, cum_ret_bm, avg_daily_ret_bm, std_daily_ret_bm, sharpe_ratio_bm, final_value_bm, portvals, portvals_bm, df_orders  = market_simulator(df_trades, df_benchmark_trades, symbol=symbol, start_val=start_val, commission=commission, impact=impact)

    plot_norm_data = plot_norm_data_vertical_lines(
                            df_orders,
                            portvals,
                            portvals_bm,
                            vert_lines=False,
                            title="Training Portfolio Value",
                            xtitle="Dates",
                            ytitle="Value ($)")


    # **** Testing ****
    start_d_test = end_d
    end_d_test =  dt.date.today() - dt.timedelta(1)


    # Get benchmark data
    benchmark_prices = fetchOnlineData(start_d_test, end_d_test, symbol)

    # Create benchmark data: Benchmark is a portfolio starting with $100,000, investing in 1000 shares of symbol and holding that position
    df_benchmark_trades_testing = create_df_benchmark(symbol, start_d_test, end_d_test, num_shares)

    df_trades_testing = stl.test_policy(symbol=symbol, start_date=start_d_test, end_date=end_d_test)

    # Retrieve performance stats via a market simulator
    test_orders_count, test_sharpe_ratio, test_cum_ret, test_std_daily_ret, test_avg_daily_ret, test_final_value, test_cum_ret_bm, test_avg_daily_ret_bm, test_std_daily_ret_bm, test_sharpe_ratio_bm, test_final_value_bm, test_portvals, test_portvals_bm, test_df_orders  = market_simulator(df_trades_testing, df_benchmark_trades_testing, symbol=symbol, start_val=start_val, commission=commission, impact=impact)

    plot_norm_data_test = plot_norm_data_vertical_lines(
                            test_df_orders,
                            test_portvals,
                            test_portvals_bm,
                            vert_lines=False,
                            title="Testing Portfolio Value",
                            xtitle="Dates",
                            ytitle="Value ($)")



    return render_template(
        # name of template
        "training.html",

        # Training data
        start_date = start_d,
        end_date = end_d,
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
        start_date_testing = start_d_test,
        end_date_testing = end_d_test,
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

    if request.method == 'POST':
        if form.validate() == False:
            flash('All fields are required.')
            return render_template('startValuesForm.html', form = form)
        else:
            return render_template('success.html')
    elif request.method == 'GET':
        return render_template('startValuesForm.html', form = form)



if __name__ == '__main__':
    app.run(host='0.0.0.0')
