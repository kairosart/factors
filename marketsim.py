"""Market simulator that processes a dataframe instead of a csv file"""

import pandas as pd
import numpy as np
import datetime as dt
#import matplotlib.pyplot as plt
from analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data
from util import normalize_data, fetchOnlineData


# Add plotly for interactive charts
from plotly.offline import init_notebook_mode, plot
#init_notebook_mode(connected=True)
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

def compute_portvals_single_symbol(df_orders, symbol, start_val=100000,
    commission=9.95, impact=0.005):
    """Compute portfolio values for a single symbol.

    Parameters:
    df_orders: A dataframe with orders for buying or selling stocks. There is
    no value for cash (i.e. 0).
    symbol: The stock symbol whose portfolio values need to be computed
    start_val: The starting value of the portfolio (initial cash available)
    commission: The fixed amount in dollars charged for each transaction
    impact: The amount the price moves against the trader compared to the
    historical data at each transaction

    Returns:
    portvals: A dataframe with one column containing the value of the portfolio
    for each trading day
    """

    # Sort the orders dataframe by date
    df_orders['Shares']  = df_orders['Shares'].apply(pd.to_numeric, downcast='float', errors='coerce')

    df_orders.sort_index(ascending=True, inplace=True)
    print(df_orders.info())

    # Get the start and end dates
    start_date = df_orders.index.min()
    end_date = df_orders.index.max()

    # Create a dataframe with adjusted close prices for the symbol and for cash
    df_prices = fetchOnlineData(start_date, end_date, symbol)
    df_prices.rename(columns={'Adj Close':symbol}, inplace=True)
    df_prices["cash"] = 1.0
    del df_prices["Symbol"]
    print("df_prices", df_prices.info())

    # Fill NAN values if any
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=True)
    df_prices.fillna(1.0, inplace=True)

    # Create a dataframe that represents changes in the number of shares by day
    df_trades = pd.DataFrame(np.zeros((df_prices.shape)), df_prices.index,
        df_prices.columns)
    print('df_trades', df_trades.info())
    for index, row in df_orders.iterrows():
        # Total value of shares purchased or sold
        traded_share_value = df_prices.loc[index, symbol] * row["Shares"]
        # Transaction cost
        transaction_cost = float(commission) + float(impact) * df_prices.loc[index, symbol] \
                            * row["Shares"]

        # Update the number of shares and cash based on the type of transaction
        # Note: The same asset may be traded more than once on a particular day
        # If the shares were bought
        if row["Shares"] > 0:
            df_trades.loc[index, symbol] = df_trades.loc[index, symbol] \
                                            + row["Shares"]
            df_trades.loc[index, "cash"] = df_trades.loc[index, "cash"] \
                                            - traded_share_value \
                                            - transaction_cost
        # If the shares were sold
        elif row["Shares"] < 0:
            df_trades.loc[index, symbol] = df_trades.loc[index, symbol] \
                                            + row["Shares"]
            df_trades.loc[index, "cash"] = df_trades.loc[index, "cash"] \
                                            - traded_share_value \
                                            - transaction_cost
    # Create a dataframe that represents on each particular day how much of
    # each asset in the portfolio
    df_holdings = pd.DataFrame(np.zeros((df_prices.shape)), df_prices.index,
        df_prices.columns)
    for row_count in range(len(df_holdings)):
        # In the first row, the number shares are the same as in df_trades,
        # but start_val must be added to cash
        if row_count == 0:
            df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1].copy()
            df_holdings.iloc[0, -1] = df_trades.iloc[0, -1] + float(start_val)
        # The rest of the rows show cumulative values
        else:
            df_holdings.iloc[row_count] = df_holdings.iloc[row_count-1] \
                                            + df_trades.iloc[row_count]
        row_count += 1

    # Create a dataframe that represents the monetary value of each asset
    df_value = df_prices * df_holdings

    # Create portvals dataframe
    portvals = pd.DataFrame(df_value.sum(axis=1), df_value.index, ["port_val"])
    return portvals
def compute_portvals_single_symbol1(df_orders, symbol, start_val=100000,
    commission=9.95, impact=0.005):
    """Compute portfolio values for a single symbol.

    Parameters:
    df_orders: A dataframe with orders for buying or selling stocks. There is
    no value for cash (i.e. 0).
    symbol: The stock symbol whose portfolio values need to be computed
    start_val: The starting value of the portfolio (initial cash available)
    commission: The fixed amount in dollars charged for each transaction
    impact: The amount the price moves against the trader compared to the
    historical data at each transaction

    Returns:
    portvals: A dataframe with one column containing the value of the portfolio
    for each trading day
    """

    # Sort the orders dataframe by date
    df_orders.sort_index(ascending=True, inplace=True)

    # Convert Shares column to float
    df_orders['Shares'] = df_orders['Shares'].astype(float)

    # Get the start and end dates
    start_date = df_orders.index.min()
    end_date = df_orders.index.max()

    # Create a dataframe with adjusted close prices for the symbol and for cash
    df_prices = fetchOnlineData(start_date, end_date, symbol)

    # Convert Adj Close column to float
    df_prices['Adj Close'] = df_prices['Adj Close'].astype(float)
    df_prices["cash"] = 1.0

    # Fill NAN values if any
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=True)
    df_prices.fillna(1.0, inplace=True)

    # Create a dataframe that represents changes in the number of shares by day
    df_trades = df_prices.copy()
    # Delete Adj Close column
    del df_trades['Adj Close']
    # Rename column symbol
    df_trades.rename(columns={'Symbol':symbol}, inplace=True)
    # Set to 0 all values of symbol column
    df_trades[symbol] = 0.0

    # Create a dataframe that represents on each particular day how much of
    # each asset in the portfolio
    df_holdings = pd.DataFrame(np.zeros((df_prices.shape)), df_prices.index,
        df_prices.columns)
    # Delete Adj Close column
    del df_holdings['Adj Close']
    # Rename column symbol
    df_holdings.rename(columns={'Symbol':symbol}, inplace=True)


    for index, row in df_orders.iterrows():
        # Total value of shares purchased or sold
        price = df_prices.loc[index, "Adj Close"]
        shares =  row["Shares"]
        traded_share_value = price * shares

        # Transaction cost
        transaction_cost = float(commission) + float(impact) * traded_share_value

        # Update the number of shares and cash based on the type of transaction
        # Note: The same asset may be traded more than once on a particular day
        # If the shares were bought
        if row["Shares"] > 0:
            df_trades.loc[index, symbol] = price + shares
            df_trades.loc[index, "cash"] = df_trades.loc[index, "cash"] \
                                            - traded_share_value \
                                            - transaction_cost
        # If the shares were sold
        elif row["Shares"] < 0:
            df_trades.loc[index, symbol] = price + shares
            df_trades.loc[index, "cash"] = df_trades.loc[index, "cash"] \
                                            - traded_share_value \
                                            - transaction_cost



    for row_count in range(len(df_holdings)):
        # In the first row, the number shares are the same as in df_trades,
        # but start_val must be added to cash
        if row_count == 0:
            df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1].copy()
            df_holdings.iloc[0, -1] = df_trades.iloc[0, -1] + float(start_val)
        # The rest of the rows show cumulative values
        else:
            df_holdings.iloc[row_count] = df_holdings.iloc[row_count-1] \
                                            + df_trades.iloc[row_count]
        row_count += 1


    # Create a dataframe that represents the monetary value of each asset
    df_value = df_prices * df_holdings

    # Create portvals dataframe
    portvals = pd.DataFrame(df_value.sum(axis=1), df_value.index, ["port_val"])
    return portvals

def market_simulator(df_orders, df_orders_benchmark, symbol, start_val=100000, commission=9.95,
    impact=0.005, daily_rf=0.0, samples_per_year=252.0):
    """
    This function takes in and executes trades from orders dataframes
    Parameters:
    df_orders: A dataframe that contains portfolio orders
    df_orders_benchmark: A dataframe that contains benchmark orders
    start_val: The starting cash in dollars
    commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    impact: The amount the price moves against the trader compared to the historical data at each transaction
    daily_rf: Daily risk-free rate, assuming it does not change
    samples_per_year: Sampling frequency per year


    Returns:
    Print out final portfolio value of the portfolio, as well as Sharpe ratio,
    cumulative return, average daily return and standard deviation of the portfolio and Benchmark.
    Plot a chart of the portfolio and benchmark performances
    """

    # Process portfolio orders
    portvals = compute_portvals_single_symbol(df_orders=df_orders, symbol=symbol,
        start_val=start_val, commission=commission, impact=impact)

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals,
     daily_rf=daily_rf, samples_per_year=samples_per_year)


    # Process benchmark orders
    portvals_bm = compute_portvals_single_symbol(df_orders=df_orders_benchmark,
        symbol=symbol, start_val=start_val, commission=commission, impact=impact)
    print('portvals_bm', portvals_bm)
    # Get benchmark stats
    cum_ret_bm, avg_daily_ret_bm, std_daily_ret_bm, sharpe_ratio_bm = get_portfolio_stats(portvals_bm,
     daily_rf=daily_rf, samples_per_year=samples_per_year)

    # Get Final values
    final_value = portvals.iloc[-1, -1]
    final_value_bm = portvals_bm.iloc[-1, -1]

    # Get orders number
    orders_count = len(portvals.index)


    # Rename columns and normalize data to the first date of the date range
    portvals.rename(columns={"port_val": "Portfolio"}, inplace=True)
    portvals_bm.rename(columns={"port_val": "Benchmark"}, inplace=True)



    return orders_count, sharpe_ratio, cum_ret, std_daily_ret, avg_daily_ret, final_value, cum_ret_bm, avg_daily_ret_bm, std_daily_ret_bm, sharpe_ratio_bm, final_value_bm, portvals, portvals_bm, df_orders
