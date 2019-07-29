1. Put all files into one folder
2. To run BestPossibleStrategy.py:
	python BestPossibleStrategy.py
	By default, it uses these parameters: testPolicy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000), and it will also produce a .png named "manual_strategy.png" to show portfolio vs benchmark line graph. 
3. To run ManualStrategy.py:
	python ManualStrategy
	By default, it uses these parameters: testPolicy(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000), and it will also produce a .png named "bps.png". 
4. To run indicators.py:
	python indicators.py
	By default, it will produce one graph for 3 indicators, and it will produce 3 .png files, one chart for each indicator.