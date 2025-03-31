import yfinance as yahooFinance
import datetime

class StockData:

    def findStockData(stockName):
        # today = datetime.date.today()
        # duration = 730
        # start = today+datetime.timedelta(-duration)
        # print(today, start)
        GetStockInformation = yahooFinance.Ticker(stockName)
        stockData = GetStockInformation.history(period="2y")
        return stockData

    def storeData(dataFrame):
        dataFrame.to_csv('StockData.csv')