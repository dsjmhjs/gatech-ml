"""MC1-P2: Optimize a portfolio.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo
import matplotlib.pyplot as plt

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    normed = prices/prices.iloc[0]
    allocGuess = np.empty(len(syms))
    allocGuess.fill(1.0/len(syms))
    bounds =  tuple((0., 1.) for _ in range(len(syms)))
    result = spo.minimize(f, allocGuess, args=(normed,), method = 'SLSQP', options={'disp': True}, bounds=bounds,
                          constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) }))
    allocs=result.x
    alloced = normed * allocs
    pos_val = alloced * 1000000
    port_val = pos_val.sum(axis=1)
    
    cum_ret = port_val[-1]/port_val[0] - 1
    daily_returns = (port_val[1:] / port_val[:-1].values) - 1
    
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_rat = np.sqrt(252) * np.mean(daily_returns - 0.0) / std_daily_ret
    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    cr, adr, sddr, sr = [cum_ret, avg_daily_ret, std_daily_ret, sharpe_rat]
    print sum(allocs)
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val/port_val[0], prices_SPY/prices_SPY[0]], keys=['Portfolio', 'SPY'], axis=1)
        plot = df_temp.plot(title="Daily portfolio value and SPY")
        plot.set_ylabel("Normalized price")
        plot.set_xlabel("Date")
        plt.savefig('plot.png')
        plt.show()

    return allocs, cr, adr, sddr, sr

def f(allocGuess, data):
    std_daily_ret = (((data * allocGuess * 1000000).sum(axis=1)[1:] / (data * allocGuess * 1000000).sum(axis=1)[:-1].values) - 1).std()
    return std_daily_ret

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
