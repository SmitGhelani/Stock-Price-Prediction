import json
from django.shortcuts import render, HttpResponse, redirect
from datetime import *
from django.contrib import messages
from stock.models import Contact
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
import nsepy
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import re
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns

graph = None
plt.style.use('ggplot')

def index(request):
    '''
    if request.user.is_authenticated:
        pass
    else:
        return redirect('/login')'''
    return render(request, 'index.html')

def about(request):
    if request.user.is_authenticated:
        pass
    else:
        return redirect('/login')
    return render(request, 'about.html')

def profile(request):
    if request.user.is_authenticated:
        pass
    else:
        return redirect('/login')

    context = {
        'uname' : request.user.username,
        'uemail' : request.user.email,
        'udate' : request.user.date_joined,
    }
    return render(request, 'profile.html',context)

def contact(request):
    if request.user.is_authenticated:
        pass
    else:
        return redirect('/login')
    if request.method == "POST":
        name = request.POST.get('name')
        age = request.POST.get('age')
        email = request.POST.get('email')
        mobino = request.POST.get('mobino')
        address = request.POST.get('address')
        date = datetime.today()
        contact = Contact(name=name, age=age, email=email, mobino=mobino, address=address, date=date)
        contact.save()
        messages.success(request, 'Form is successfully submitted.')
    return render(request, 'contact.html')

def login2(request):
    print(request)
    if request.method == "POST":
        login_user = request.POST.get('login_username')
        login_password = request.POST.get('login_password')
        user = authenticate(username=login_user, password=login_password)
        if user is not None:
            print("login")
            login(request, user)
            messages.success(request, 'Login Successfully.')
            return redirect('/index')
        else:
            messages.success(request, 'Invalid Username or Password')
            return render(request, 'login.html')
    return render(request, 'login.html')


def pre(request):
    if request.user.is_authenticated:
        pass
    else:
        return redirect('/login')
    if request.method == "POST":
        symbol_name = request.POST.get('symbol_name')
        duration = request.POST.get('dura')
        if all(i.isdigit() for i in duration):
            today = date.today()
            start = today + timedelta(-int(duration))
            stockData = nsepy.get_history(symbol=symbol_name, start=start, end=today)
            stockData = stockData.reset_index()
            stockData['Date'] = stockData['Date'].astype(str)
            stockData['Open'] = stockData['Open'].astype(float)
            stockData['Volume'] = stockData['Volume'].astype(float)
            chart = get_plot(stockData['Open'].tolist(), stockData['Volume'].tolist())
            stockData = stockData.reindex(index=stockData.index[::-1])
            json_records = stockData.reset_index().to_json(orient='records')
            data = json.loads(json_records)
            context = {'d': data,
                       'duration': duration,
                       'sym_name': symbol_name,
                       'visible': "visible",
                       'chart': chart}
            return render(request, 'pre.html', context)
        else:
            messages.success(request, 'Please enter duration in integer !')
    context = {'visible': "invisible"}
    return render(request, 'pre.html', context)

def logout2(request):
    logout(request)
    return redirect('/login')

def register(request):
    if request.user.is_authenticated:
        return redirect('/')
    else:
        pass
    if request.method == "POST":
        reg_user = request.POST.get('reg_username')
        reg_email = request.POST.get('reg_email')
        reg_password = request.POST.get('reg_password')
        user = User.objects.create_user(reg_user, reg_email, reg_password)
        special_characters = """!@#$%^&*()-+?_=,<>/"""
        regex = '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
        if any(i.isdigit() for i in reg_user) or any(c in special_characters for c in reg_user):
            messages.success(request, 'Username must contain alphabet only !')
            if (re.search(regex, reg_email)):
                pass
            else:

                return redirect('/register')

            return redirect('/register')
        #messages.success(request, 'Email must be in valid format !')
        user.save()
        messages.success(request, 'You have already registered !')
        return redirect('/login')
    return render(request, 'register.html')

def forecast(request):
    global graph
    if request.user.is_authenticated:
        pass
    else:
        return redirect('/login')
    if request.method == "POST":
        stock_name = request.POST.get('stock_name')
        pred_df, df = calPred(stock_name)
        graph = None
        plot1(df, pred_df, "Stock Price Prediction of "+str(stock_name), 'Date', 'Price', 'blue')
        print(pred_df.reset_index().columns)

        pred_df = pred_df.reset_index()
        pred_df['index'] = pred_df['index'].astype(str)
        json_records = pred_df.to_json(orient='records')
        data = json.loads(json_records)

        df = df.reset_index()
        df['Date'] = df['Date'].astype(str)
        df['Open'] = df['Open'].astype(float)
        df = df.iloc[::-1]
        json_records2 = df.head(120).to_json(orient='records')
        data2 = json.loads(json_records2)

        context = {
            'chart': graph,
            'd' : data,
            'd2' : data2,
            'visible': "visible",
        }
        #print('GG :: ',graph)
        return render(request, 'forecast.html', context)
    context = {'visible': "invisible"}
    return render(request, 'forecast.html',context)

def calPred(stkname):
    df = StockData.findStockData(stkname)
    num_data = len(df)
    df = df.reset_index()
    copy_df = pd.DataFrame(df)
    dt = df['Date']
    df = df[['Close', 'Volume']]

    pred_len = int(120)
    df['Prediction'] = df[['Close']].shift(-pred_len)

    x = np.array(df.drop(['Prediction'], 1))
    x = preprocessing.scale(x)

    x_pred = x[-pred_len:]
    x = x[:-pred_len]

    y = np.array(df['Prediction'])
    y = y[:-pred_len]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    clf = LinearRegression(learning_rate=1)
    clf.train(x_train, y_train)

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
    copy_df = copy_df.set_index('Date')
    pred_df = pd.DataFrame(pred_df['Prediction'])
    return pred_df, copy_df


def plot1(org_df, pred_df, ttl="", x_label='x', y_label='y', color='blue'):
    global graph
    plt.clf()
    sns.set_theme(style="darkgrid")
    org_df['Close'].plot(figsize=(15, 6), color=color)
    pred_df['Prediction'].plot(figsize=(15, 6), color='orange')
    plt.legend(loc=4)
    set_labels(ttl, x_label, y_label)
    graph = get_graph()

def set_labels(ttl, x_label, y_label):

    plt.title(ttl)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    #plt.show()



class StockData:

    def findStockData(stockName):
        today = datetime.date.today()
        duration = 720
        start = today + datetime.timedelta(-duration)

        stockData = nsepy.get_history(symbol=stockName, start=start, end=today)
        return stockData

    def storeData(dataFrame) -> object:
        dataFrame.to_csv('StockData.csv')


class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.slope = None
        self.c = None

    def train(self, x, y):
        n_samples, n_features = x.shape
        self.slope = np.zeros(n_features)
        self.c = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(x, self.slope) + self.c
            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.slope -= self.lr * dw
            self.c -= self.lr * db

    def predict(self, x):
        y_approx = np.dot(x, self.slope) + self.c
        return y_approx

def get_graph():
    buffer = BytesIO()
    buffer.flush()
    plt.savefig(buffer, format='png')  #,dpi=700
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    print("print",buffer)
    buffer.close()
    return graph


def get_plot(x, y):
    plt.switch_backend('AGG')
    plt.figure(figsize=(10, 5))
    plt.title("Graph")
    plt.plot(x, y)
    plt.xlabel('ABC')
    plt.ylabel('DFG')
    plt.tight_layout()
    graph = get_graph()
    return graph
