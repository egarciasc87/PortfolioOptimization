import pandas as pd
import numpy as np
from pandas_datareader import data as web
from datetime import datetime
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

plt.style.use("fivethirtyeight")

#get the tickers
assets = ["FB", "AMZN", "AAPL", "NFLX", "GOOG"]

#assign weights to the stocks
weights = np.array([0.2] * 5)

#get the portfolio a starting date
stockStartDate = '2013-01-01'

#get the portfolio ending date
stockEndDate = datetime.today().strftime('%Y-%m-%d')
stockEndDate = '2020-03-16'

#create a dataframe to sotre the adjusted close price of stocks
df = pd.DataFrame()
tempDf = pd.DataFrame()

for ticker in assets:
    fileName = ticker + "_Week" + ".csv"
    tempDf = pd.read_csv(fileName)
    tempDf = tempDf.drop(columns=["Open", "High", "Low", "Vol.", "Change %"])
    tempDf.columns = ["Date", ticker]
    tempDf = tempDf.set_index("Date")
    print(tempDf)

    if len(df.columns) == 0:
        df = tempDf
    else:
        df = df.join(tempDf)

#df = df.sort_values("Date")
print(df)

#visually show stock/portfolio
title = "Portfolio Close price History"
my_stocks = df

for c in my_stocks.columns.values:
    plt.plot(my_stocks[c], label = c)

plt.title(title)
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Close Price USD", fontsize = 18)
plt.legend(my_stocks.columns.values, loc="upper left")
#plt.show()

#Get return (weekly, daily)
returns = df.pct_change()
returns = returns
print(returns)

#create annualized covariance matrix
cov_matrix_annual = returns.cov() * 252
print(cov_matrix_annual)

cov_matrix_annual = returns.cov() * 52
print(cov_matrix_annual)

#calculate portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
print(port_variance)

#calculate portfolio volatility (standard deviation)
port_volatility = np.sqrt(port_variance)
print(port_volatility)

#annual portfolio return
portfolio_simple_annual_return = np.sum(returns.mean() * weights) * 252
print(portfolio_simple_annual_return)

#show expected annual return, volatility (risk), variance
percent_var = str(round(port_variance, 4) * 100) + "%"
percent_vol = str(round(port_volatility, 4) * 100) + "%"
percent_ret = str(round(portfolio_simple_annual_return, 4) * 100) + "%"
print("Expected annual return: ", percent_ret)
print("Annual Volatility: ", percent_vol)
print("Annual Variance: ", percent_var)
