# Xicheng Huang xhuang343
import pandas as pd
import numpy as np
import datetime as dt
import os
import util

def compute_portvals(df_trades, symbol, sd, ed, startval, market_impact=0.0, commission_cost=0.0):
    """Simulate the market for the given date range and orders file."""
    orders_df = pd.DataFrame(columns=['Shares','Order','Symbol'])
    for row_idx in df_trades.index:
        nshares = df_trades.loc[row_idx][0]
        if nshares == 0:
            continue
        order = 'sell' if nshares < 0 else 'buy'
        new_row = pd.DataFrame([[abs(nshares),order,symbol],],columns=['Shares','Order','Symbol'],index=[row_idx,])
        orders_df = orders_df.append(new_row)
    symbols = []
    orders = []
    orders_df = orders_df.sort_index()
    for date, order in orders_df.iterrows():
        shares = order['Shares']
        action = order['Order']
        symbol = order['Symbol']
        if action.lower() == 'sell':
            shares *= -1
        order = (date, symbol, shares)
        orders.append(order)
        symbols.append(symbol)
    symbols = list(set(symbols))
    dates = pd.date_range(sd, ed)
    prices_all = util.get_data(symbols, dates)
    prices = prices_all[symbols]
    prices = prices.fillna(method='ffill').fillna(method='bfill')
    prices['_CASH'] = 1.0
    trades = pd.DataFrame(index=prices.index, columns=symbols)
    trades = trades.fillna(0)
    cash = pd.Series(index=prices.index)
    cash = cash.fillna(0)
    cash.ix[0] = startval
    for date, symbol, shares in orders:
        price = prices[symbol][date]
        val = shares * price
        # transaction cost model
        val += commission_cost + (pd.np.abs(shares)*price*market_impact)
        positions = prices.ix[date] * trades.sum()
        totalcash = cash.sum()
        if (date < prices.index.min()) or (date > prices.index.max()):
            continue
        trades[symbol][date] += shares
        cash[date] -= val
    trades['_CASH'] = cash
    holdings = trades.cumsum()
    df_portvals = (prices * holdings).sum(axis=1)
    return df_portvals


def author():
    return 'xhuang343'

if __name__ == "__main__":
    test_code()
