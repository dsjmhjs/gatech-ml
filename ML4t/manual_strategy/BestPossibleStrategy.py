import os
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
import datetime as dt 
import numpy as np
import marketsimcode as mkt

if os.getenv('DISPLAY') is None:
    plt.switch_backend('agg')

def save_or_show(plt, filename):
    if os.getenv('DISPLAY') is None:
        print "No $DISPLAY detected. Writing file '{}'".format(filename)
        plt.savefig(filename)
    else:
        plt.show()

def testPolicy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
	symbols = []
	symbols.append(symbol)
	stock_prices = get_data(symbols, pd.date_range(sd, ed), addSPY=False)
	stock_prices = stock_prices.dropna()
	price_diff = pd.DataFrame(index=stock_prices.index)
	price_diff['price_diff'] = stock_prices[:-1] - stock_prices[1:].values
	# price_diff = price_diff.shift(-1)
	price_diff['trades'] = 0
	if price_diff.iloc[0, 0] < 0:
		price_diff.iloc[0,1] = 1000
	elif price_diff.iloc[0, 0] > 0:
		price_diff.iloc[0,1] = -1000
	else:
		price_diff.iloc[0,1] = 0
	price_diff.iloc[1:,1][price_diff.iloc[1:,0]< 0] = 2000
	price_diff.iloc[1:,1][price_diff.iloc[1:,0]> 0] = -2000
	price_diff[(price_diff.shift(1)['trades']<0) & (price_diff['trades']<0)] = 0
	price_diff[(price_diff.shift(1)['trades']>0) & (price_diff['trades']>0)] = 0
	df_trades = pd.DataFrame(data=price_diff.iloc[:,1])
	df_trades['Order']='NONE'
	df_trades['Symbol']=symbol
	df_trades.loc[df_trades['trades']>0,'Order']='BUY'
	df_trades.loc[df_trades['trades']<0,'Order']='SELL'
	df_trades['trades'] = df_trades['trades'].abs()
	port_val = mkt.compute_portvals(df_trades, sv, 0.0, 0.0)
	#calculate benchmark
	df_trades_bm = pd.DataFrame(index=df_trades.index)
	df_trades_bm['Symbol']='JPM'
	df_trades_bm['trades']=0
	df_trades_bm['Order']=''
	df_trades_bm.iloc[0,1]=1000
	df_trades_bm.iloc[0,2]='BUY'
	port_val_bm = mkt.compute_portvals(df_trades_bm, sv, 0.0, 0.0)
	assess_performance(port_val, port_val_bm, True)
	return df_trades

def assess_performance(port_val, port_val_bm, gen_plot=False):
	cum_ret = port_val[-1]/port_val[0] - 1
	daily_returns = (port_val[1:] / port_val[:-1].values) - 1
	avg_daily_ret = daily_returns.mean()
	std_daily_ret = daily_returns.std()
	print "Volatility (stdev of daily returns):", std_daily_ret
	print "Average Daily Return:", avg_daily_ret
	print "Cumulative Return:", cum_ret
	print "End Value:", port_val[-1]

	cum_ret_bm = port_val_bm[-1]/port_val_bm[0] - 1
	daily_returns_bm = (port_val_bm[1:] / port_val_bm[:-1].values) - 1
	avg_daily_ret_bm = daily_returns_bm.mean()
	std_daily_ret_bm = daily_returns_bm.std()
	print "Benchmark Volatility (stdev of daily returns):", std_daily_ret_bm
	print "Benchmark Average Daily Return:", avg_daily_ret_bm
	print "Benchmark Cumulative Return:", cum_ret_bm
	print "Benchmark End Value:", port_val_bm[-1]
	if gen_plot:
		df_temp = pd.concat([port_val/port_val[0], port_val_bm/port_val_bm[0]], keys=['Portfolio', 'JPM'], axis=1)
		plot = df_temp.plot(title="Best Possible Strategy", color=['Black', 'Blue'])
		plot.set_ylabel("Normalized price")
		plot.set_xlabel("Date")
		save_or_show(plt, 'bps.png')
	
if __name__ == "__main__":
	testPolicy()