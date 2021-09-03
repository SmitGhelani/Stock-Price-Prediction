import nsepy
import datetime
import pymysql
from pandas.io import sql

class StockData:

    def findStockData(stockName):
        today = datetime.date.today()
        duration = 730
        start = today+datetime.timedelta(-duration)

        stockData = nsepy.get_history(symbol=stockName, start=start,end=today)
        return stockData

    def storeData(dataFrame):
        dataFrame.to_csv('StockData.csv')