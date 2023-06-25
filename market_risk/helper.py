import pandas as pd
import yfinance as yf
import numba as nb
from math import exp

def get_data(list_of_stocks: list, start_date:str, end_date:str , columns:list, interval:str, resample=None) -> pd.DataFrame:
    lst=[]
    for stock in list_of_stocks:
        equity=yf.Ticker(stock)
        equity_data=equity.history(start=start_date, end=end_date, interval=interval)
        if resample != None:
            equity_data = equity_data.resample(resample).last()
        equity_data['returns']=equity_data['Close'].pct_change()
        a=equity_data['returns']
        lst.append(a)
    df = pd.concat(lst, axis=1, join= 'outer')
    df.columns= columns
    df.dropna(axis=0,inplace=True)
    return df

get_data_nb = nb.jit(get_data)

def half_kelly_criterion(data: pd.DataFrame, f:float) -> pd.Series:
    equ = "equity_{:.2f}".format(f)
    cap = "capital_{:.2f}".format(f)
    data[equ] = 1
    data[cap] = data.equ * f
    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[i]
        try:
            data.loc[t, cap] = data[cap].loc[t_1] * data["our_portfolio"].loc[t]
            data.loc[t, equ] = data[cap].loc[t] - data[cap].loc[t_1] + data[equ].loc[t_1]
            data.loc[t, cap] = data[equ].loc[t] * f
        except Exception:
            print("Please run the previous cells")
            
    

def losses(p : pd.DataFrame) -> pd.Series:
    """
    :param p: portfolio time series
    :return: portfolio losses (returns) in absolute value
    """
    returns = p.pct_change()
    return returns.where(returns < 0).dropna().abs()

def gains(p: pd.DataFrame) -> pd.Series:
    """
    :param p: portfolio time series
    :return: portfolio gains (returns)
    """
    returns = p.pct_change()
    return returns.where(returns > 0).dropna()

def non_parametric_VAR(p:pd.Series, q:float) -> float:
    """
    :param p: portfolio
    :param q: quantile
    :return: Value at risk of the portfolio
    """
    return losses(p).quantile(c_i)

def non_parametric_ES(p: pd.DataFrame, q: float):
    """
    :param p: portfolio
    :param q: quantile
    :return: Conditional value at risk E(X1| X1 <= VAR(X))
    """
    loss = losses(p)
    return loss.where(loss >= historical_VAR(p, q)).dropna().mean()