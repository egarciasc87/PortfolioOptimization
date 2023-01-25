import pandas as pd
import numpy as np
import pandas_datareader as pdr
import yfinance as yf
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
    print("Annual Volatility: ", percent_vol)
    print("Expected annual return: ", percent_ret)
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


def get_tickers(exchange):
    listStocks = []
    ticker = "X"

    if exchange != "BVL":
        while ticker != "":
            ticker = input("Enter ticker: ")

            if ticker != "":
                listStocks.append(ticker)
    else:
        listStocks = ["ALICORC1", "IFS", "MINSURI1", "FERREYC1", "INRETC1"]

    return listStocks


def get_stock_prices(exchange):
    start_date = "2017-01-01"
    end_date = "2022-12-31"
    df = pd.DataFrame()
    listStocks = get_tickers(exchange)

    for stock in listStocks:
        if exchange == "BVL":
            tempDf = pd.read_csv("BVL_" + stock + ".csv")
            tempDf = tempDf.drop(columns=["Apertura", "Cierre",
                "Máximo", "Mínimo", "Promedio", "Cantidad negociada",
                "Monto negociado", "Fecha anterior"])
            tempDf.columns = ["Date", stock]
            tempDf = tempDf.set_index("Date")
        else:
            tempDf = yf.download(stock, start_date, end_date)
            #tempDf = web.DataReader(stock, start_date, end_date)
            tempDf = tempDf.drop(columns=["Open", "High",
                "Low", "Close", "Volume"])
            tempDf.columns = [stock]
            #print(tempDf)

        if len(df.columns) == 0:
            df = tempDf
        else:
            df = df.join(tempDf)

    return df, listStocks


def convert_percentage(value):
    percentage = str(value) + "%"
    return percentage


def method_2(exchange):
    #1. get data
    #df = pd.DataFrame()
    df, listStocks = get_stock_prices(exchange)
    #df = df.iloc[::-1]
    print(df.head())


    #2. get returns
    nr_observations = len(df)
    nr_trading_days_year = 252
    daily_returns = df.pct_change()
    #daily_returns = df.pct_change().apply(lambda x: np.log(1+x))
    daily_returns = daily_returns #* 100
    print("\nDaily return:\n")
    print(daily_returns.head())
    #print(daily_returns.iloc[0:10])
    #print((daily_returns * 100).iloc[0:10])

    #3. expected return, std, variance, covariance
    temp = daily_returns
    #temp = temp.reset_index()
    #temp["Month"] = pd.to_datetime(temp["Date"]).dt.month
    #print(temp.head())
    temp = temp.groupby(by=[temp.index.year, temp.index.month]).mean()

    for stock in listStocks:
        temp["Monthly_" + stock] = temp[stock] + 1

    #print(len(temp))
    print(temp)

    yearly_return = daily_returns.mean() * nr_trading_days_year * 100
    average_return = daily_returns.mean()
    standar_deviation = daily_returns.std() * 100
    variance = daily_returns.var()
    sharpe_ratio = yearly_return/standar_deviation
    covariance = daily_returns.cov()
    correlation = daily_returns.corr()

    #print(monthly_mean_return)
    print("\nExpected yearly return:\n", yearly_return)
    print("\nStandar deviation:\n", standar_deviation)
    print("\nVariance:\n", variance)
    print("\nSharpe ratio:\n", sharpe_ratio)
    #print("\nCovariance:\n", covariance)
    #print("\nCorrelation:\n", correlation)

    #4. equally distributed portfolio
    portfolio_weight = [1.0/len(listStocks)] * len(listStocks)
    assets = pd.concat([yearly_return, standar_deviation], axis=1)
    assets.columns = ["Returns", "Volatility"]
    print("\nRisk vs Return -> Port. equially distributed:\n", assets)

    weighted_yearly_ret = yearly_return.T * portfolio_weight
    print("\nWeighted yearly return:\n", weighted_yearly_ret)
    #print("\nDaily meanreturn:\n", daily_returns.mean())
    calculate_efficient_frontier(listStocks, yearly_return, covariance, df)


def calculate_efficient_frontier(listStocks, yearly_return, covariance, df):
    #5- create different portfolio variations
    portfolio_returns = [] #pd.DataFrame(columns=listStocks)
    portfolio_volatilities = [] #pd.DataFrame(columns=listStocks)
    portfolio_weights = [] #pd.DataFrame(columns=listStocks)
    portfolio_sharpe_ratio = []
    nr_portfolios = 25000
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
    print("\nPortfolios:\n", portfolios.head())
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
method_2("NN")
