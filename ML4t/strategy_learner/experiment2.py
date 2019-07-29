# Xicheng Huang xhuang343
import os
import matplotlib.pyplot as plt
from StrategyLearner import StrategyLearner
import marketsimcode as mkt
import datetime as dt
import ManualStrategy as ms
import pandas as pd
import numpy as np

if os.getenv('DISPLAY') is None:
    plt.switch_backend('agg')

def save_or_show(plt, filename):
    if os.getenv('DISPLAY') is None:
        print "No $DISPLAY detected. Writing file '{}'".format(filename)
        plt.savefig(filename)
    else:
        plt.show()


if __name__=="__main__":

	impacts=np.arange(0.0, 0.055, 0.005)
	numOfTrades=[]
	cumReturns = []
	stdDailyRets=[]
	for i in impacts:
		learner = StrategyLearner(impact=i)
		learner.addEvidence(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
		df_trades = learner.testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000)
		port_val = mkt.compute_portvals(df_trades, list(df_trades)[0], dt.datetime(2008,1,1), dt.datetime(2009,12,31), 100000, 0.0, i)
		numOfTrades.append(len(df_trades[df_trades['JPM']!=0]))
		
		cum_ret = port_val[-1]/port_val[0] - 1
		daily_returns = (port_val[1:] / port_val[:-1].values) - 1
		avg_daily_ret = daily_returns.mean()
		std_daily_ret = daily_returns.std()
		cumReturns.append(cum_ret)
		stdDailyRets.append(std_daily_ret)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(impacts, numOfTrades)
	plt.xticks(impacts)
	ax.set(title='Experiment 2 - Impact vs. Number of Trades', ylabel='Number of Trades', xlabel='Impact')
	save_or_show(plt, "experiment2_numOfTrades.png")

	fig2 = plt.figure()
	ax = fig2.add_subplot(111)
	ax.plot(impacts, cumReturns)
	plt.xticks(impacts)
	ax.set(title='Experiment 2 - Impact vs. Cumulative Return', ylabel='Cumulative Return', xlabel='Impact')
	save_or_show(plt, "experiment2_cumRet.png")

	fig3 = plt.figure()
	ax = fig3.add_subplot(111)
	ax.plot(impacts, stdDailyRets)
	plt.xticks(impacts)
	ax.set(title='Experiment 2 - Impact vs. Standard Deviation of Daily Return', ylabel='SD. of Daily Return', xlabel='Impact')
	save_or_show(plt, "experiment2_std.png")