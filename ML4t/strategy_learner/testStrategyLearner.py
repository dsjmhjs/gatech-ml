from StrategyLearner import StrategyLearner
import marketsimcode as mkt
import datetime as dt


def assess_performance(port_val, df_trades, gen_plot=False):
	cum_ret = port_val[-1]/port_val[0] - 1
	daily_returns = (port_val[1:] / port_val[:-1].values) - 1
	avg_daily_ret = daily_returns.mean()
	std_daily_ret = daily_returns.std()
	print "Volatility (stdev of daily returns):", std_daily_ret
	print "Average Daily Return:", avg_daily_ret
	print "Cumulative Return:", cum_ret
	print "End Value:", port_val[-1]


if __name__=="__main__":
	learner = StrategyLearner()
	learner.addEvidence(symbol = "JPM")
	df_trades = learner.testPolicy(symbol = "JPM")
	port_val = mkt.compute_portvals(df_trades, list(df_trades)[0], dt.datetime(2008,1,1), dt.datetime(2009,12,31), 100000, 0.0, 0.0)
	assess_performance(port_val, df_trades, True)
	# print "======================================================"
	# df_trades = learner.testPolicy( symbol = "SINE_FAST_NOISE", \
 #        sd=dt.datetime(2010,1,1), \
 #        ed=dt.datetime(2011,12,31), \
 #        sv = 100000)
	# port_val = mkt.compute_portvals(df_trades, list(df_trades)[0], 100000, 0.0, 0.0)
	# assess_performance(port_val, df_trades, True)



	# cum_ret_bm = port_val_bm[-1]/port_val_bm[0] - 1
	# daily_returns_bm = (port_val_bm[1:] / port_val_bm[:-1].values) - 1
	# avg_daily_ret_bm = daily_returns_bm.mean()
	# std_daily_ret_bm = daily_returns_bm.std()
	# print "Benchmark Volatility (stdev of daily returns):", std_daily_ret_bm
	# print "Benchmark Average Daily Return:", avg_daily_ret_bm
	# print "Benchmark Cumulative Return:", cum_ret_bm
	# print "Benchmark End Value:", port_val_bm[-1]
	# if gen_plot:
	# 	df_temp = pd.concat([port_val/port_val[0], port_val_bm/port_val_bm[0]], keys=['Portfolio', 'Benchmark'], axis=1)
	# 	plot = df_temp.plot(title="Manual Strategy", color=['Black', 'Blue'])
	# 	buy_dates = df_trades[df_trades['Order']=='BUY'].index
	# 	for d in buy_dates:
	# 		plt.axvline(x=d, color='green')
	# 	sell_dates = df_trades[df_trades['Order']=='SELL'].index
	# 	for d in sell_dates:
	# 		plt.axvline(x=d, color='red')
	# 	plot.set_ylabel("Normalized price")
	# 	plot.set_xlabel("Date")
	# 	save_or_show(plt, 'manual_strategy.png')