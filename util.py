"""Util functions for StrategyLearner."""

import datetime as dt
import os
import pandas as pd
import numpy as np
import csv

# To fetch data
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from sklearn.preprocessing import MinMaxScaler

yf.pdr_override()

def symbol_to_path(symbol, base_dir=None):
    """Return CSV file path given ticker symbol."""
    if base_dir is None:
        base_dir = os.environ.get("MARKET_DATA_DIR", './data/')
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbol, colname = 'Adj Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""

    df = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                     parse_dates=True, usecols=['Date', colname], na_values=['nan'])
    df = df.rename(columns={colname: symbol})

    # Fill NAN values if any
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    df.fillna(1.0, inplace=True)
    return df



def slice_df(df_to_slice, dates):
    # Slice the Data
    from_date = pd.to_datetime(dates.values[0])
    to_date = pd.to_datetime(dates.values[-1])
    df_slice = df_to_slice.loc[from_date:to_date, :]
    return df_slice



def df_to_cvs(df, symbol):
    outname = symbol + '.csv'
    outdir = './data'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fullname = os.path.join(outdir, outname)
    df.to_csv(fullname)

def fetchOnlineData(dt_start, symbol):
    # Add a day to dt_end for Yahoo purpose
    dt_end = dt.date.today() - dt.timedelta(1)

    # Get data of trading days between the start and the end.
    df = pdr.get_data_yahoo(
            # tickers list (single tickers accepts a string as well)
            tickers = symbol,
            # start date (YYYY-MM-DD / datetime.datetime object)
            # (optional, defaults is 1950-01-01)
            start = dt_start,
            # end date (YYYY-MM-DD / datetime.datetime object)
            # (optional, defaults is Today)
            end = dt_end,
            # return a multi-index dataframe
            # (optional, default is Panel, which is deprecated)
            as_panel = False,
            # group by ticker (to access via data['SPY'])
            # (optional, default is 'column')
            group_by = 'ticker',
            # adjust all OHLC automatically
            # (optional, default is False)
            auto_adjust = False
    )
    if len(df.index):
        df_to_cvs(df, symbol)
        return True
    else:
        return False




def get_orders_data_file(basefilename):
    return open(os.path.join(os.environ.get("ORDERS_DATA_DIR",'orders/'),basefilename))

def get_learner_data_file(basefilename):
    return open(os.path.join(os.environ.get("LEARNER_DATA_DIR",'Data/'),basefilename),'r')

def get_robot_world_file(basefilename):
    return open(os.path.join(os.environ.get("ROBOT_WORLDS_DIR",'testworlds/'),basefilename))


def normalize_data(df):
    """Normalize stock prices using the first row of the dataframe"""
    return df/df.iloc[0,:]

def scaling_data(df, column):
    # Scaling column
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df

def compute_daily_returns(df):
    """Compute and return the daily return values"""
    daily_returns = df.pct_change()
    daily_returns.iloc[0,:] = 0
    return daily_returns


def compute_sharpe_ratio(k, avg_return, risk_free_rate, std_return):
    """
    Compute and return the Sharpe ratio
    Parameters:
    k: adjustment factor, sqrt(252) for daily data, sqrt(52) for weekly data, sqrt(12) for monthly data
    avg_return: daily, weekly or monthly return
    risk_free_rate: daily, weekly or monthly risk free rate
    std_return: daily, weekly or monthly standard deviation
    Returns:
    sharpe_ratio: k * (avg_return - risk_free_rate) / std_return
    """
    return k * (avg_return - risk_free_rate) / std_return



def load_txt_data(dirpath, filename):
    """ Load the data from a txt file and store them as a numpy array

    Parameters:
    dirpath: The path to the directory where the file is stored
    filename: The name of the file in the dirpath

    Returns:
    np_data: A numpy array of the data
    """

    try:
        filepath= os.path.join(dirpath, filename)
    except KeyError:
        print ("The file is missing")

    np_data = np.loadtxt(filepath, dtype=str)

    return np_data


def get_exchange_days(start_date = dt.datetime(1964,7,5), end_date = dt.datetime(2020,12,31),
    dirpath = "../data/dates_lists", filename="NYSE_dates.txt"):
    """ Create a list of dates between start_date and end_date (inclusive) that correspond
    to the dates there was trading at an exchange. Default values are given based on NYSE.

    Parameters:
    start_date: First timestamp to consider (inclusive)
    end_date: Last day to consider (inclusive)
    dirpath: The path to the directory where the file is stored
    filename: The name of the file in the dirpath

    Returns:
    dates: A list of dates between start_date and end_date on which an exchange traded
    """

    # Load a text file located in dirpath
    dates_str = load_txt_data(dirpath, filename)
    all_dates_frome_file = [dt.datetime.strptime(date, "%m/%d/%Y") for date in dates_str]
    df_all_dates = pd.Series(index=all_dates_frome_file, data=all_dates_frome_file)

    selected_dates = [date for date in df_all_dates[start_date:end_date]]

    return selected_dates


def get_data_as_dict(dates, symbols, keys):
    """ Create a dictionary with types of data (Adj Close, Volume, etc.) as keys. Each value is
    a dataframe with symbols as columns and dates as rows

    Parameters:
    dates: A list of dates of interest
    symbols: A list of symbols of interest
    keys: A list of types of data of interest, e.g. Adj Close, Volume, etc.

    Returns:
    data_dict: A dictionary whose keys are types of data, e.g. Adj Close, Volume, etc. and
    values are dataframes with dates as indices and symbols as columns
    """

    data_dict = {}
    for key in keys:
        df = pd.DataFrame(index=dates)
        for symbol in symbols:
            df_temp = pd.read_csv(symbol_to_path(symbol), index_col="Date",
                    parse_dates=True, usecols=["Date", key], na_values=["nan"])
            df_temp = df_temp.rename(columns={key: symbol})
            df = df.join(df_temp)
        data_dict[key] = df
    return data_dict

def create_df_benchmark(df, num_shares):
    """Create a dataframe of benchmark data. Benchmark is a portfolio consisting of
    num_shares of the symbol in use and holding them until end_date.
    """
    # Get adjusted close price data
    benchmark_prices = df
    # Create benchmark df: buy num_shares and hold them till the last date
    df_benchmark_trades = pd.DataFrame(
        data=[(benchmark_prices.index.min(), num_shares),
        (benchmark_prices.index.max(), -num_shares)],
        columns=["Date", "Shares"])
    df_benchmark_trades.set_index("Date", inplace=True)
    return df_benchmark_trades

def create_df_trades(orders, symbol, num_shares, cash_pos=0, long_pos=1, short_pos=-1):
    """Create a dataframe of trades based on the orders executed. +1000
    indicates a BUY of 1000 shares, and -1000 indicates a SELL of 1000 shares.
    """
    # Remove cash positions to make the "for" loop below run faster
    non_cash_orders = orders[orders != cash_pos]
    trades = []
    for date in non_cash_orders.index:
        if non_cash_orders.loc[date] == long_pos:
            trades.append((date, num_shares))
        elif non_cash_orders.loc[date] == short_pos:
            trades.append((date, -int(num_shares)))
    df_trades = pd.DataFrame(trades, columns=["Date", "Shares"])
    df_trades.set_index("Date", inplace=True)
    return df_trades


def get_nasdaq_tickers():
    os.system("curl --ftp-ssl anonymous:jupi@jupi.com "
              "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt "
              "> nasdaq.lst")
    os.system("tail -n +9 nasdaq.lst | cat | sed '$d' | sed 's/|/ /g' > "
          "nasdaq.lst2")
    os.system("awk '{print $1}' nasdaq.lst2 > nasdaq.csv")
    os.system("echo; head nasdaq.csv; echo '...'; tail nasdaq.csv")
    os.system("cp nasdaq.csv ./data/nasdaq.csv")

if __name__=="__main__":
    get_nasdaq_tickers()