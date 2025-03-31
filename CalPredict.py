import numpy as np
import datetime
import Graph as plt
from PredictionAlgorithm import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from StockData import StockData as stkdt
import pandas as pd

def calPred(stkname):
    df = stkdt.findStockData(stkname)
    print(df)
    num_data = len(df)
    df = df.reset_index()
    copy_df=pd.DataFrame(df)
    dt=df['Date']
    df = df[['Close','Volume']]

    pred_len = int(120)
    df['Prediction'] = df[['Close']].shift(-pred_len)

    x = np.array(df.drop(['Prediction'], axis=1))
    x = preprocessing.scale(x)

    x_pred = x[-pred_len:]
    x = x[:-pred_len]

    y = np.array(df['Prediction'])
    y = y[:-pred_len]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    clf = LinearRegression(learning_rate=0.01, n_iters=1000)
    clf.train(x_train, y_train)

    y_pred = clf.predict(x_test)
    print("accurecy:",mean_squared_error(y_test, y_pred))
    price_prediction = clf.predict(x_pred)

    df.dropna(inplace=True)
    df['Prediction'] = np.nan

    last_date = pd.to_datetime(dt.iloc[-1])
    last_sec = last_date.timestamp()
    one_day_sec = 86400
    next_sec = last_sec + one_day_sec

    for i in price_prediction:
        next_date = datetime.datetime.fromtimestamp(next_sec)
        datetime.datetime.fromtimestamp(next_sec)
        next_sec += 86400
        df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

    df = pd.DataFrame(df)
    pred_df = df[-120:]
    copy_df=copy_df.set_index('Date')
    pred_df=pd.DataFrame(pred_df['Prediction'])
    return pred_df,copy_df

pred_df,df= calPred('META')
stkdt.storeData(df)
# print(pred_df.head(), df.head())
plt.plot(df,pred_df,"Stock Price Prediction of META",'Date','Price', 'blue')