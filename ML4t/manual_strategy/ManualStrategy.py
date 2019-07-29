import os
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data, plot_data
import datetime as dt 
import numpy as np
import marketsimcode as mkt
import indicators as ic

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
	stock_prices = get_data(symbols, pd.date_range(sd, ed), addSPY=False).dropna()
	#get trades by indicators
	bollinger_trades = ic.bollinger_bands_indicator(symbol, pd.date_range(sd, ed))
	cci_trades = ic.commodity_channel_index_indicator(symbol, pd.date_range(sd, ed))
	golden_cross_trades = ic.golden_cross(symbol, pd.date_range(sd, ed))
	golden_cross_long_trades = ic.golden_cross(symbol, pd.date_range(sd, ed), short_window=50, long_window=200)
	#combine trades
	trade_frames=[bollinger_trades[bollinger_trades['trades']!=0], cci_trades[cci_trades['trades']!=0], golden_cross_trades[golden_cross_trades['trades']!=0], golden_cross_long_trades[golden_cross_long_trades['trades']!=0]]
	#trade_frames=[bollinger_trades[bollinger_trades['trades']!=0]]
	df_trades=pd.concat(trade_frames)
	df_trades = df_trades.sort_index()
	df_trades.loc[(df_trades.shift(1)['trades']<0) & (df_trades['trades']<0), 'trades'] = 0
	df_trades.loc[(df_trades.shift(1)['trades']>0) & (df_trades['trades']>0), 'trades'] = 0
	df_trades['trades'] = df_trades['trades'].abs()
	df_trades.ix[0, 'trades']=1000
	df_trades.loc[df_trades['trades']==0, 'Order']='NONE'
	df_trades[symbol] = df_trades['trades']
	port_val = mkt.compute_portvals(df_trades, 100000, 0.0, 0.0)
	#calculate benchmark
	df_trades_bm = pd.DataFrame(index=stock_prices.index)
	df_trades_bm['Symbol']='JPM'
	df_trades_bm['trades']=0
	df_trades_bm['Order']=''
	df_trades_bm.iloc[0,1]=1000
	df_trades_bm.iloc[0,2]='BUY'
	port_val_bm = mkt.compute_portvals(df_trades_bm, 100000, 9.95, 0.005)
	assess_performance(port_val, port_val_bm, df_trades, True)
	print df_trades[symbol]
	return df_trades

def assess_performance(port_val, port_val_bm, df_trades, gen_plot=False):
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
		df_temp = pd.concat([port_val/port_val[0], port_val_bm/port_val_bm[0]], keys=['Portfolio', 'Benchmark'], axis=1)
		plot = df_temp.plot(title="Manual Strategy", color=['Black', 'Blue'])
		buy_dates = df_trades[df_trades['Order']=='BUY'].index
		for d in buy_dates:
			plt.axvline(x=d, color='green')
		sell_dates = df_trades[df_trades['Order']=='SELL'].index
		for d in sell_dates:
			plt.axvline(x=d, color='red')
		plot.set_ylabel("Normalized price")
		plot.set_xlabel("Date")
		save_or_show(plt, 'manual_strategy.png')
	
if __name__ == "__main__":
	testPolicy()