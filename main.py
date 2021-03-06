import io
import os

from alpha_vantage.techindicators import TechIndicators
from flask import Flask, render_template, session, request, flash
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features

from forecastPrice import showforcastpricesvalues
from form import StartValuesForm, get_tickers, pricesForecast
import pandas as pd
import datetime as dt
from markupsafe import Markup
import fix_yahoo_finance as yf
yf.pdr_override()

from util import create_df_benchmark, fetchOnlineData, get_data, \
    symbol_to_path, df_to_cvs, get_data_av, scaling_data, normalize_data, scaling_series, slice_df, get_data_av_min
from strategyLearner import strategyLearner
from marketsim import market_simulator
from indicators import get_momentum, get_sma, get_rolling_mean, get_rolling_std, get_bollinger_bands, \
    get_RSI, plot_cum_return, plot_momentum, plot_sma_indicator, plot_rsi_indicator, \
    plot_stock_prices, plot_bollinger, plot_norm_data_vertical_lines


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

    # Get lookback date
    lookback_date = request.form.get('loookback_date', type=str)

    # Validate lookback date
    d1 = dt.datetime.strptime(lookback_date, '%m/%d/%Y')
    d2 = dt.datetime.today() - dt.timedelta(days=30)
    date_diff_ok = d1 < d2

    if date_diff_ok:
        lookback_date = dt.datetime.strptime(lookback_date, '%m/%d/%Y')
        start_d = f"{lookback_date:%Y-%m-%d %H:%M:%S}"
        yesterday = dt.date.today() - dt.timedelta(1)

        # Import data from Alpha Vantage
        dates = pd.date_range(start_d, dt.date.today())
        portf_value = get_data_av(symbol, dates, del_cols=True)

        # Import data from Yahoo
        #portf_value = fetchOnlineData(start_d, symbol, dt.date.today())
    else:
        return render_template('error.html',
                                form='showvalues',
                                error="Lookback date must be at least 30 days earlier today.")






    # Session variables
    session['start_val'] = request.form['start_val']
    session['symbol'] = symbol
    session['start_d'] = start_d
    session['num_shares'] = request.form['num_shares']
    session['commission'] = request.form['commission']
    session['impact'] = request.form['impact']

    # **** Calculating indicators ****
    # Rename 'Adj Close' to symbol
    portf_value.rename(columns={'Adj Close': symbol}, inplace=True)
    stl = strategyLearner()
    df_norm = stl.get_features1(portf_value, symbol, print=True)

    # **** Plotting indicators ****

    ##### PRICE MOVEMENTS ####
    plot_prices = plot_stock_prices(portf_value[symbol], symbol)

    #### MOMENTUM ####

    plot_mom = plot_momentum(df_norm, symbol)
    #### ROLLILNGER BANDS ####
    plot_boll = plot_bollinger(df_norm, symbol)


    ##### SMA ####
    plot_sma = plot_sma_indicator(df_norm, symbol)

    ##### RSI ####
    plot_rsi = plot_rsi_indicator(df_norm, symbol)

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

# Training
@app.route('/benchmark', methods=['POST', 'GET'])
def training():
    # **** Training ***
    # Getting session variables
    start_d = str(session.get('start_d', None))
    start_val = int(session.get('start_val', None))
    symbol = session.get('symbol', None)
    num_shares = int(session.get('num_shares', None))
    commission = float(session.get('commission', None))
    impact = float(session.get('impact', None))


    # Create a dataframe from csv file
    #df = get_data(symbol)
    # Import data from Alpha Vantage
    dates = pd.date_range(start_d, dt.date.today())
    df = get_data_av(symbol, dates, del_cols=True)

    # Rename column Adj Close
    df.rename(columns={'Close': symbol}, inplace=True)

    if not isinstance(df, pd.DataFrame):
        return render_template(
            # name of template
            "error.html",
            # now we pass in our variables into the template
            form='showvalues', error="Data source can't be reached at this moment. Try again.")

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
                          num_states=3000, num_actions=3, dyna=200)

    epochs, cum_returns, trades, portvals, df_features, thresholds  = stl.add_evidence(df_training,
                                           symbol,
                                           start_val=start_val,
                                           start_date=start_date_training,
                                           end_date=end_date_training)
    df_trades = stl.test_policy(df_training,
                                symbol,
                                start_date=start_date_training,
                                end_date=end_date_training)

    # cum_returns = map(lambda cum_returns: cum_returns * 100, cum_returns)
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

    #TODO Check test results

    # **** Testing ****
    # Out-of-sample or testing period: Perform similiar steps as above,
    # except that we don't train the data (i.e. run add_evidence again)

    df_benchmark_trades = create_df_benchmark(df_testing, num_shares)
    df_trades = stl.test_policy(df_testing, symbol, start_date=start_date_testing,
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
        start_date=start_date_training,
        end_date=end_date_training,
        symbol=symbol,
        div_placeholder_plot_cum=Markup(plot_cum),
        div_placeholder_plot_norm_data=Markup(plot_norm_data),
        orders_count_p=orders_count,
        sharpe_ratio_p=round(sharpe_ratio, 3),
        cum_ret_p=round(cum_ret, 3),
        std_daily_ret_p=round(std_daily_ret, 3),
        avg_daily_ret_p=round(avg_daily_ret, 3),
        final_value_p=round(final_value, 3),
        sharpe_ratio_b=round(sharpe_ratio_bm, 3),
        cum_ret_b=round(cum_ret_bm, 3),
        std_daily_ret_b=round(std_daily_ret_bm, 3),
        avg_daily_ret_b=round(avg_daily_ret_bm, 3),
        final_value_b=round(final_value_bm, 3),

        # Testing datasets
        start_date_testing=start_date_testing,
        end_date_testing=end_date_testing,
        div_placeholder_plot_norm_data_testing=Markup(plot_norm_data_test),
        orders_count_p_testing=test_orders_count,
        sharpe_ratio_p_testing=round(test_sharpe_ratio, 3),
        cum_ret_p_testing=round(test_cum_ret, 3),
        std_daily_ret_p_testing=round(test_std_daily_ret, 3),
        avg_daily_ret_p_testing=round(test_avg_daily_ret, 3),
        final_value_p_testing=round(test_final_value, 3),
        sharpe_ratio_b_testing=round(test_sharpe_ratio_bm, 3),
        cum_ret_b_testing=round(test_cum_ret_bm, 3),
        std_daily_ret_b_testing=round(test_std_daily_ret_bm, 3),
        avg_daily_ret_b_testing=round(test_avg_daily_ret_bm, 3),
        final_value_b_testing=round(test_final_value_bm, 3)

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
            return render_template('error.html')
    elif request.method == 'GET':

        return render_template('startValuesForm.html',
                               form =form,
                               data=tickers)

# Initial form to get values for price forecasting
@app.route('/showforecastform', methods = ['GET', 'POST'])
def showforecastform():
    form = pricesForecast()

    # Get ticker name list
    tickers = get_tickers('nasdaq_tickers_name')

    if request.method == 'POST':
        result = request.form
        # Get symbol
        symbol = request.form.get('ticker_select', type=str)

        # Get Forecast date
        forecast_date = request.form.get('forecastDate', type=str)
        forecast_date = dt.datetime.strptime(forecast_date, '%m/%d/%Y')

        # Get Forecast model
        forecast_model = request.form.get('model_Selection', type=str)

        # Get Forecast time
        forecast_time = request.form.get('forecast_Time', type=str)

        # Get values from forecast_time
        end = None
        forecast_time = int(forecast_time[7:end])


        # Get Lookback
        forecast_lookback = request.form.get('look_Back', type=int)

        # Get lookback date of data to train and test
        start_d = forecast_date - dt.timedelta(forecast_lookback)
        start_d = f"{start_d:%Y-%m-%d}"
        #yesterday = dt.date.today() - dt.timedelta(1)
        dates = pd.date_range(start_d, dt.date.today())
        # Import data from Yahoo
        if result['model_Selection'] == 'model1' or result['model_Selection'] == 'model2':
            #portf_value = fetchOnlineData(start_d, symbol, yesterday, del_cols=False)
            portf_value = get_data_av(symbol, dates, del_cols=False)
        else:
            #portf_value = fetchOnlineData(start_d, symbol, yesterday)
            portf_value = get_data_av(symbol, dates)

        if not isinstance(portf_value, pd.DataFrame):
            return render_template(
                # name of template
                "error.html",
                # now we pass in our variables into the template
                form='showforecastform',
                error="Error downloading data from Yahoo. Try again",
            )


        # XGBoost model
        if result['model_Selection'] == 'model1':

            symbol, start_d, forecast_date,  plot_prices_pred, daily_return_percentage = showforcastpricesvalues(symbol, portf_value, forecast_model,
                                                                                   forecast_time, start_d, forecast_date,
                                                                                   forecast_lookback)

            final_forecast_day = dt.date.today() + dt.timedelta(forecast_time)
            final_forecast_day = f"{final_forecast_day:%Y-%m-%d}"
            return render_template(
                # name of template
                "forecastPrices.html",
                # now we pass in our variables into the template
                symbol=symbol,
                forecast_date=forecast_date.strftime("%Y-%m-%d"),
                forecast_model_name="Decision Tree XGBoost",
                forecast_time=forecast_time,
                forecast_lookback=forecast_lookback,
                forecast_final_date=final_forecast_day,
                daily_return_percentage=Markup(daily_return_percentage.to_html(classes="table-sm")),
                div_placeholder_stock_prices_pred=Markup(plot_prices_pred),
                titles=['na', 'Stock Prices '],
                model='XGBoost',
            )

        # KNN model
        elif result['model_Selection'] == 'model2':

            symbol, start_d, forecast_date,  plot_prices_pred, daily_return_percentage = showforcastpricesvalues(symbol, portf_value, forecast_model,
                                                                                   forecast_time, start_d, forecast_date,
                                                                                   forecast_lookback)

            final_forecast_day = dt.date.today() + dt.timedelta(forecast_time)
            final_forecast_day = f"{final_forecast_day:%Y-%m-%d}"
            return render_template(
                # name of template
                "forecastPrices.html",
                # now we pass in our variables into the template
                symbol=symbol,
                forecast_date=forecast_date.strftime("%Y-%m-%d"),
                forecast_model_name="KNN",
                forecast_time=forecast_time,
                forecast_lookback=forecast_lookback,
                forecast_final_date=final_forecast_day,
                daily_return_percentage=Markup(daily_return_percentage.to_html(classes="table-sm")),
                div_placeholder_stock_prices_pred=Markup(plot_prices_pred),
                titles=['na', 'Stock Prices '],
                model='KNN',
            )

        # ARIMA Model
        elif result['model_Selection'] == 'model3':
            symbol, start_d, forecast_date, plot_prices_pred, daily_return_percentage = showforcastpricesvalues(symbol, portf_value,
                                                                                                 forecast_model,
                                                                                                 forecast_time, start_d,
                                                                                                 forecast_date,
                                                                                                 forecast_lookback)
            final_forecast_day = dt.date.today() + dt.timedelta(forecast_time)
            final_forecast_day = f"{final_forecast_day:%Y-%m-%d}"
            return render_template(
                # name of template
                "forecastPrices.html",
                # now we pass in our variables into the template
                symbol=symbol,
                forecast_date=forecast_date.strftime("%Y-%m-%d"),
                forecast_model_name='ARIMA',
                forecast_time=forecast_time,
                forecast_lookback=forecast_lookback,
                forecast_final_date=final_forecast_day,
                daily_return_percentage=Markup(daily_return_percentage.to_html(classes="table-sm")),
                div_placeholder_stock_prices_pred=Markup(plot_prices_pred),
                titles=['na', 'Stock Prices '],
                model='ARIMA',
            )
        # LSTM Model
        elif result['model_Selection'] == 'model4':
            symbol, start_d, forecast_date, plot_prices_pred, daily_return_percentage = showforcastpricesvalues(symbol, portf_value, forecast_model,
                                                                                   forecast_time, start_d, forecast_date,
                                                                                   forecast_lookback)
            final_forecast_day = dt.date.today() + dt.timedelta(forecast_time)
            final_forecast_day = f"{final_forecast_day:%Y-%m-%d}"
            return render_template(
                # name of template
                "forecastPrices.html",
                # now we pass in our variables into the template
                symbol=symbol,
                forecast_date=forecast_date.strftime("%Y-%m-%d"),
                forecast_model_name='LSTM',
                forecast_time=forecast_time,
                forecast_lookback=forecast_lookback,
                forecast_final_date=final_forecast_day,
                daily_return_percentage=Markup(daily_return_percentage.to_html(classes="table-sm")),
                div_placeholder_stock_prices_pred=Markup(plot_prices_pred),
                titles=['na', 'Stock Prices '],
                model='LSTM',
            )

    elif request.method == 'GET':

        return render_template('pricesForecastForm.html',
                               form=form,
                               data=tickers)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
