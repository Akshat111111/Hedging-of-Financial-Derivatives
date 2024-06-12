Dataset link: https://finance.yahoo.com/quote/TTM/history?p=TTM
This dataset contains all the previous record of stock price of Tata Motors Company.

To use the dataset in the code you can use this command
 
import yfinance as yf
import datetime as dt
df = yf.download('TTM', dt.datetime(2012,1,1) , dt.datetime(2023,1,7)) 