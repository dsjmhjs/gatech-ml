"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders_df.sort_index(inplace=True)
    start_date = orders_df.index.values[0]
    end_date = orders_df.index.values[-1]
    indices = pd.date_range(start_date, end_date)
    portvals = pd.DataFrame(np.full(indices.shape[0], start_val), index=indices)
    stock_prices = get_data(orders_df.Symbol.unique().tolist(), pd.date_range(start_date, end_date), addSPY=False)
    portvals['Cash'] = start_val
    for index, row in orders_df.iterrows():
    	portvals.loc[portvals.index>=index,'Cash']-=commission
    	portvals.loc[portvals.index>=index,'Cash']-=row['Shares'] * stock_prices.loc[index, row['Symbol']] * impact
    	if row['Order'] == 'BUY':
    		if row['Symbol'] not in portvals:
    			portvals[row['Symbol']]=0
    		portvals.loc[portvals.index>=index, row['Symbol']] += row['Shares']
    		portvals.loc[portvals.index>=index,'Cash'] -= row['Shares'] * stock_prices.loc[index, row['Symbol']]
    	elif row['Order'] == 'SELL':
    		if row['Symbol'] not in portvals:
    			portvals[row['Symbol']]=0
    		portvals.loc[portvals.index>=index, row['Symbol']] -= row['Shares']
    		portvals.loc[portvals.index>=index,'Cash'] += row['Shares'] * stock_prices.loc[index, row['Symbol']]
    portvals['HoldingsVal'] = (portvals.iloc[:,2:] * stock_prices).sum(axis=1, skipna=True)
    portvals[0] = portvals['HoldingsVal'] + portvals['Cash']
    return portvals[0].dropna()

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-10.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv, commission=9.95, impact=0.005)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

def author():
    return 'xhuang343'

if __name__ == "__main__":
    test_code()
