"""Bollinger Bands."""
# Xicheng Huang xhuang343
import os
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
import datetime as dt 
import numpy as np

if os.getenv('DISPLAY') is None:
    plt.switch_backend('agg')

def save_or_show(plt, filename):
    if os.getenv('DISPLAY') is None:
        print "No $DISPLAY detected. Writing file '{}'".format(filename)
        plt.savefig(filename)
    else:
        plt.show()


def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling(window=window).mean()


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return values.rolling(window=window).std()


def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band


def bollinger_bands_indicator(symbol='SPY', dates = pd.date_range('2012-01-01', '2012-12-31'), window=21, show_plot=False):
    # Read data
    
    symbols = []
    symbols.append(symbol)
    df = get_data(symbols, dates, addSPY=False)
    df = df.dropna()
    df=df/df.iloc[0]
    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm = get_rolling_mean(df[symbol], window)
    # 2. Compute rolling standard deviation
    rstd = get_rolling_std(df[symbol], window)

    # 3. Compute upper and lower bands
    upper_band, lower_band = get_bollinger_bands(rm, rstd)
    if show_plot:
        # Plot raw values, rolling mean and Bollinger Bands
        ax = df[symbol].plot(title="Bollinger Bands", label=symbol)
        rm.plot(label='Rolling mean', ax=ax)
        upper_band.plot(label='upper band', ax=ax)
        lower_band.plot(label='lower band', ax=ax)

        # Add axis labels and legend
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='lower right')
        save_or_show(plt, 'bbi.png')

    df_trades = pd.DataFrame(index=dates)
    df_trades['Symbol']=symbol
    df_trades['trade']=False
    df_trades['Order']='NONE'
    df_trades['trades']=0
    previous_prices = df[symbol].shift(-1)
    indices = df[((previous_prices > upper_band) & (df[symbol] < upper_band))].index
    df_trades = df_trades[~df_trades.index.duplicated(keep='first')]
    df_trades.loc[indices, 'trade']=True
    df_trades.loc[indices, 'Order']='BUY'
    df_trades.loc[indices, 'trades']=2000
    indices2 = df[((previous_prices < lower_band) & (df[symbol] > lower_band))].index
    df_trades.loc[indices2, 'trade']=True
    df_trades.loc[indices2, 'Order']='SELL'
    df_trades.loc[indices2, 'trades']=-2000
    df_trades['bbi_upper_band'] = upper_band
    df_trades['bbi_lower_band'] = lower_band
    return df_trades

def commodity_channel_index_indicator(symbol='SPY', dates = pd.date_range('2012-01-01', '2012-12-31'), window=21, show_plot=False):
    symbols = []
    symbols.append(symbol)
    df = get_data(symbols, dates, addSPY=False)
    df = df.dropna()
    df=df/df.iloc[0]
    # Compute Bollinger Bands
    # 1. Compute rolling mean
    rm = get_rolling_mean(df[symbol], window)

    # 2. Compute rolling standard deviation
    rstd = get_rolling_std(df[symbol], window)

    commodity_channel_index = (df[symbol] - rm) / (0.015 * rstd)
    # Plot raw values, rolling mean and Bollinger Bands
    if show_plot:
        plt.figure(1)
        plt.subplot(211)
        ax = df[symbol].plot(title="Commodity Channel Index", label=symbol)
        ax.set_xticklabels([])
        rm.plot(label='Rolling mean', ax=ax)
        plt.subplot(212)
        ax2 = commodity_channel_index.plot(label='CCI')
        plt.axhline(y=100, color='r')
        plt.axhline(y=0)
        plt.axhline(y=-100, color='g')

        # Add axis labels and legend
        ax2.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax2.set_ylabel("CCI")
        ax.legend(loc='lower right')
        ax2.legend(loc='lower left')
        save_or_show(plt, 'cci.png')
    # calculate trades
    df_trades = pd.DataFrame(index=df.index)
    df_trades['Symbol']=symbol
    df_trades['trade']=False
    df_trades['Order']='NONE'
    df_trades['trades']=0
    commodity_channel_index=commodity_channel_index.dropna()
    previous_commodity_channel_index = commodity_channel_index.shift(1)
    indices = commodity_channel_index[((previous_commodity_channel_index > 100) & (commodity_channel_index < 100))].index
    df_trades = df_trades[~df_trades.index.duplicated(keep='first')]
    df_trades.loc[indices, 'trade']=True
    df_trades.loc[indices, 'Order']='SELL'
    df_trades.loc[indices, 'trades']=-2000
    indices = commodity_channel_index[((previous_commodity_channel_index < -100) & (commodity_channel_index > -100))].index
    df_trades.loc[indices, 'trade']=True
    df_trades.loc[indices, 'Order']='BUY'
    df_trades.loc[indices, 'trades']=2000
    df_trades['cci'] = commodity_channel_index
    return df_trades

def golden_cross(symbol='SPY', dates = pd.date_range('2012-01-01', '2012-12-31'), short_window=15, long_window=50, show_plot=False):
    symbols = []
    symbols.append(symbol)
    df = get_data(symbols, dates, addSPY=False)
    df = df.dropna()
    df=df/df.iloc[0]
    # Compute Bollinger Bands
    # 1. Compute rolling mean
    short_rm = get_rolling_mean(df[symbol], short_window)
    long_rm = get_rolling_mean(df[symbol], long_window)
    # Plot raw values, rolling mean and Bollinger Bands
    previous_short = short_rm.shift(-1)
    previous_long = long_rm.shift(-1)
    crossing = (((short_rm <= long_rm) & (previous_short >= previous_long))
            | ((short_rm >= long_rm) & (previous_short <= previous_long)))
    crossing_dates = long_rm[crossing].index
    if show_plot:
        plt.figure(2)
        ax = df[symbol].plot(title="Golden Cross", label=symbol)
        short_rm.plot(label='Short-Term {}-day Moving Average'.format(short_window), ax=ax)
        long_rm.plot(label='Long-Term {}-day Moving Average'.format(long_window), ax=ax)
        plt.plot(crossing_dates, short_rm[crossing_dates], 'ro', label='Cross Over')
        # Add axis labels and legend
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='lower right')
        save_or_show(plt, 'golden_cross.png')
    # calculate trades
    df_trades = pd.DataFrame(index=df.index)
    df_trades = df_trades[~df_trades.index.duplicated(keep='first')]
    df_trades['Symbol']=symbol
    df_trades['trade']=False
    df_trades['Order']='NONE'
    df_trades['trades']=0
    df_trades.loc[long_rm[((short_rm <= long_rm) & (previous_short >= previous_long))].index, 'Order'] = 'SELL'
    df_trades.loc[long_rm[((short_rm <= long_rm) & (previous_short >= previous_long))].index, 'trades'] = -2000
    df_trades.loc[long_rm[((short_rm >= long_rm) & (previous_short <= previous_long))].index, 'Order'] = 'BUY'
    df_trades.loc[long_rm[((short_rm >= long_rm) & (previous_short <= previous_long))].index, 'trades'] = 2000
    df_trades.loc[long_rm[((short_rm <= long_rm) & (previous_short >= previous_long))].index, 'trade'] = True
    df_trades.loc[long_rm[((short_rm >= long_rm) & (previous_short <= previous_long))].index, 'trade'] = True
    df_trades['gc_short_rm'] = short_rm
    df_trades['gc_long_rm'] = long_rm

    return df_trades

if __name__ == "__main__":
    bollinger_bands_indicator(symbol='JPM', dates = pd.date_range('2008-01-01', '2009-12-31'), show_plot=True)
    golden_cross(symbol='JPM', dates = pd.date_range('2008-01-01', '2009-12-31'), show_plot=True, short_window=15, long_window=50)
    commodity_channel_index_indicator(symbol='JPM', dates = pd.date_range('2008-01-01', '2009-12-31'), show_plot=True)

