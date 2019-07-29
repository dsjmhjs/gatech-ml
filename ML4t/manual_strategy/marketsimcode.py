import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_df, start_val = 1000000, commission=9.95, impact=0.005):
    start_date = orders_df.index.values[0]
    end_date = orders_df.index.values[-1]
    stock_prices = get_data(orders_df.Symbol.unique().tolist(), pd.date_range(start_date, end_date), addSPY=False)
    stock_prices = stock_prices.dropna()
    portvals = pd.DataFrame(np.full(stock_prices.index.shape[0], start_val), index=stock_prices.index)
    portvals['Cash'] = start_val
    for index, row in orders_df.iterrows():
        if row['trades'] != 0:
        	portvals.loc[portvals.index>=index,'Cash']-=commission
        	portvals.loc[portvals.index>=index,'Cash']-=row['trades'] * stock_prices.loc[index, row['Symbol']] * impact
        	if row['Order'] == 'BUY':
        		if row['Symbol'] not in portvals:
        			portvals[row['Symbol']]=0
        		portvals.loc[portvals.index>=index, row['Symbol']] += row['trades']
        		portvals.loc[portvals.index>=index,'Cash'] -= row['trades'] * stock_prices.loc[index, row['Symbol']]
        	elif row['Order'] == 'SELL':
        		if row['Symbol'] not in portvals:
        			portvals[row['Symbol']]=0
        		portvals.loc[portvals.index>=index, row['Symbol']] -= row['trades']
        		portvals.loc[portvals.index>=index,'Cash'] += row['trades'] * stock_prices.loc[index, row['Symbol']]
    portvals['HoldingsVal'] = (portvals.iloc[:,2:] * stock_prices).sum(axis=1, skipna=True)
    portvals[0] = portvals['HoldingsVal'] + portvals['Cash']
    # with pd.option_context('display.max_rows', None, 'display.max_columns', 5):
    #     print portvals
    return portvals[0].dropna()


def author():
    return 'xhuang343'

if __name__ == "__main__":
    test_code()
