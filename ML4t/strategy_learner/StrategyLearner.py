"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""
# Xicheng Huang xhuang343
import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import random
import indicators
from BagLearner import BagLearner
from RTLearner import RTLearner

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.learner = BagLearner(learner=RTLearner,kwargs={"leaf_size":5},bags=500,boost=False,verbose=False)

    def getIndicatorsValues(self, symbol, sd, ed):
    	dates = pd.date_range(sd, ed)
    	prices = ut.get_data([symbol], dates, addSPY=False)
    	prices = prices.dropna()
        bbi_trades = indicators.bollinger_bands_indicator(symbol, dates)
        cci_trades = indicators.commodity_channel_index_indicator(symbol, dates)
        gc_trades = indicators.golden_cross(symbol, dates)
        gc_long_trades = indicators.golden_cross(symbol, dates, short_window=50, long_window=200)
        combined_trades = pd.DataFrame(index=prices.index)
        # combined_trades['trade'] = (bbi_trades['trade'] | cci_trades['trade'] | gc_trades['trade'] | gc_long_trades['trade'])
        combined_trades=pd.concat([combined_trades, bbi_trades.ix[:,-2:]], axis=1, join_axes=[combined_trades.index])
        combined_trades=pd.concat([combined_trades, cci_trades.ix[:,-1:]], axis=1, join_axes=[combined_trades.index])
        combined_trades=pd.concat([combined_trades, gc_trades.ix[:,-2:]], axis=1, join_axes=[combined_trades.index])
        combined_trades=pd.concat([combined_trades, gc_long_trades.ix[:,-2:]], axis=1, join_axes=[combined_trades.index])
        return combined_trades

    def addEvidence(self, symbol = "SINE_FAST_NOISE", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 100000): 

    	combined_trades = self.getIndicatorsValues(symbol, sd, ed)
        # trade_days = combined_trades[combined_trades['trade']].index
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices = ut.get_data(syms, dates, addSPY=False)
        n_days = -15
        # trades = pd.DataFrame(index=trade_days)
        # trades['ret'] = (prices.ix[trades.index + pd.DateOffset(n_days)].values / prices.ix[trades.index]) - 1.0
        trades = pd.DataFrame(index=prices.dropna().index)
        trades['ret'] = prices.shift(n_days) / prices - 1.0
        ybuy = 0.05
        ysell = -0.05
        if self.impact > 0.0:
        	ybuy += self.impact*2
        	ysell -= self.impact*2
        combined_trades['trades']=0
        combined_trades.loc[trades[trades['ret']>ybuy].index, 'trades'] = 1
        combined_trades.loc[trades[trades['ret']<ysell].index, 'trades'] = -1
        combined_trades = combined_trades.dropna()
        self.learner.addEvidence(combined_trades.iloc[:,1:-1].as_matrix(), combined_trades.iloc[:,-1].as_matrix())

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "SINE_FAST_NOISE", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 100000):
    	combined_trades = self.getIndicatorsValues(symbol, sd, ed)
    	pred_trades = self.learner.query(combined_trades.iloc[:,1:].as_matrix())
    	dates = pd.date_range(sd, ed)
    	df_trades = pd.DataFrame(index=combined_trades.index)
    	df_trades[symbol] = 0
    	df_trades.values[:,:]=0
    	df_trades.values[np.where(pred_trades == 1), :] = 2000
    	df_trades.values[np.where(pred_trades == -1), :] = -2000
    	df_trades = df_trades.drop(df_trades[df_trades[symbol]==0].index)
    	if len(df_trades) > 0:
	    	if df_trades[df_trades[symbol]!=0].iloc[0,0] == 2000:
	    		df_trades.loc[df_trades[df_trades[symbol]!=0].index[0]] = 1000
	    	else:
	    		df_trades.loc[df_trades[df_trades[symbol]!=0].index[0]] = -1000
    	df_trades.loc[((df_trades.shift(1)[symbol]<0) & (df_trades[symbol]<0)), symbol] = 0
    	df_trades.loc[((df_trades.shift(1)[symbol]>0) & (df_trades[symbol]>0)), symbol] = 0
    	return df_trades

if __name__=="__main__":
    print "One does not simply think up a strategy"
