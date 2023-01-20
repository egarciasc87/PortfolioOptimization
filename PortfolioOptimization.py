import pandas as pd
import numpy as np
from pandas_datareader import data as web
from datetime import datetime
import matplotlib.pyplot as plt

import math
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

plt.style.use("fivethirtyeight")

def method_1():
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
        #fileName = ticker + ".csv"
        tempDf = pd.read_csv(fileName)
        tempDf = tempDf.drop(columns=["Open", "High", "Low", "Vol.", "Change %"])
        tempDf.columns = ["Date", ticker]
        tempDf = tempDf.set_index("Date")
        #tempDf = tempDf.sort_index(axis=0, ascending=False)
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
    #print(cov_matrix_annual)

    cov_matrix_annual = returns.cov() * 52
    print(cov_matrix_annual)

    #calculate portfolio variance
    port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
    print(port_variance)

    #calculate portfolio volatility (standard deviation)
    port_volatility = np.sqrt(port_variance)
    print(port_volatility)

    #annual portfolio return
    portfolio_simple_annual_return = np.sum(returns.mean() * weights) * 52
    print(portfolio_simple_annual_return)

    #Case 1: show expected annual return, volatility (risk), variance
    percent_var = str(round(port_variance, 4) * 100) + "%"
    percent_vol = str(round(port_volatility, 4) * 100) + "%"
    percent_ret = str(round(portfolio_simple_annual_return, 4) * 100) + "%"
    print("Expected annual return: ", percent_ret)
    print("Annual Volatility: ", percent_vol)
    print("Annual Variance: ", percent_var)


    #Calculate expected return and
    #annualized sample covariance matrix
    print("\n")
    mu = expected_returns.mean_historical_return(df)
    s = risk_models.sample_cov(df)

    #optimize for maximun sharp ratio
    ef = EfficientFrontier(mu, s)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    print(cleaned_weights)
    ef.portfolio_performance(verbose = True)

    lates_prices = get_latest_prices(df)
    weights = cleaned_weights
    da = DiscreteAllocation(weights, lates_prices, total_portfolio_value = 15000)
    allocation, leftover = da.lp_portfolio()
    print("Dicrete allocation:", allocation)
    print("Funds remaining: ${:.2f}".format(leftover))


def method_2():
    #1. get data
    #
    listStocks = ["ALICORC1", "IFS", "MINSURI1", "FERREYC1", "INRETC1"]
    df = pd.DataFrame()

    for stock in listStocks:
        tempDf = pd.read_csv("BVL_" + stock + ".csv")
        tempDf = tempDf.drop(columns=["Apertura", "Cierre",
            "Máximo", "Mínimo", "Promedio", "Cantidad negociada",
            "Monto negociado", "Fecha anterior"])
        tempDf.columns = ["Date", stock]
        tempDf = tempDf.set_index("Date")

        if len(df.columns) == 0:
            df = tempDf
        else:
            df = df.join(tempDf)

    df = df.iloc[::-1]
    print(df)

    #2. get returns
    nr_observations = len(df)
    nr_trading_days_year = 252
    daily_returns = df.pct_change()
    #daily_returns = df.pct_change().apply(lambda x: np.log(1+x))
    daily_returns = daily_returns #* 100
    print("\n")
    print(daily_returns)
    #print(daily_returns.iloc[0:10])
    #print((daily_returns * 100).iloc[0:10])

    #3. expected return, std, variance, covariance
    yearly_return = daily_returns.mean() * nr_trading_days_year
    average_return = daily_returns.mean()
    standar_deviation = daily_returns.std()
    variance = daily_returns.var()
    sharpe_ratio = average_return/standar_deviation
    covariance = daily_returns.cov()
    correlation = daily_returns.corr()

    # yearly_return = pd.DataFrame(yearly_return)
    # yearly_return.columns = ["Value"]
    #
    # standar_deviation = pd.DataFrame(standar_deviation)
    # standar_deviation.columns = ["Value"]
    #
    # variance = pd.DataFrame(variance)
    # variance.columns = ["Value"]
    #
    # sharpe_ratio = pd.DataFrame(sharpe_ratio)
    # sharpe_ratio.columns = ["Value"]

    print("\nExpected yearly return:\n", yearly_return.T)
    print("\nStandar deviation:\n", standar_deviation.T)
    print("\nVariance:\n", variance.T)
    print("\nSharpe ratio:\n", sharpe_ratio.T)
    print("\nCovariance:\n", covariance)
    #print("\nCorrelation:\n", correlation)

    #4. portfolio weights
    portfolio_weight = [1.0/len(listStocks)] * len(listStocks)
    assets = pd.concat([yearly_return, standar_deviation], axis=1)
    assets.columns = ["Returns", "Volatility"]
    #print(portfolio_weight)

    weighted_yearly_ret = yearly_return.T * portfolio_weight
    print("\nWeighted yearly return:\n", weighted_yearly_ret)
    print("\nReturn VS Volatility:\n", assets)

    #5- create different portfolio variations
    portfolio_returns = [] #pd.DataFrame(columns=listStocks)
    portfolio_volatilities = [] #pd.DataFrame(columns=listStocks)
    portfolio_weights = [] #pd.DataFrame(columns=listStocks)
    portfolio_sharpe_ratio = []
    nr_portfolios = 5000
    nr_assets = len(listStocks)
    index = 0

    for x in range(nr_portfolios):
        weights = np.random.random(nr_assets)
        weights = weights/np.sum(weights)
        portfolio_weights.append(weights)
        returns = np.dot(weights, yearly_return)
        portfolio_returns.append(returns)
        var = covariance.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        sd = np.sqrt(var)
        annual_sd = np.sqrt(var * 250)
        portfolio_volatilities.append(annual_sd)
        sharpe_ratio = returns / annual_sd
        portfolio_sharpe_ratio.append(sharpe_ratio)

    #print(portfolio_volatilities)
    #print(portfolio_returns)
    #print(portfolio_weights)
    data = {"Returns": portfolio_returns, "Volatility": portfolio_volatilities, "Sharpe Ratio": portfolio_sharpe_ratio}

    for counter, symbol in enumerate(df.columns.tolist()):
        data[symbol + " weight"] = [w[counter] for w in portfolio_weights]


    portfolios = pd.DataFrame(data)
    print("\nPortfolios:\n", portfolios)
    #portfolios.plot.scatter(x="Volatility", y="Returns", marker="o", s=10, alpha=0.3, grid=True, figsize=[10,10])
    #portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])

    plt.scatter("Volatility", "Returns", data=portfolios)
    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    plt.show()

    #minimun Volatility
    min_volatility = portfolios.iloc[portfolios["Volatility"].idxmin()]
    max_return = portfolios.iloc[portfolios["Returns"].idxmax()]
    max_sharpe = portfolios.iloc[portfolios["Sharpe Ratio"].idxmax()]
    print("\nMin volatility:\n", min_volatility)
    print("\nMax return:\n", max_return)
    print("\nMax sharpe:\n", max_sharpe)


#method_1()
method_2()
