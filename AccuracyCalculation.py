import CalPredict
from StockData import StockData as stkdt
import Graph as plt

pred_df,df= CalPredict.calPred('META')
stkdt.storeData(df)
# print(pred_df.head(), df.head())
plt.plot(df,pred_df,"Stock Price Prediction of META",'Date','Price', 'blue')