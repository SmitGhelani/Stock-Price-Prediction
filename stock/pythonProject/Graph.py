import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot(org_df,pred_df,ttl="",  x_label='x', y_label='y',color='blue'):
    org_df['Close'].plot(figsize=(15, 6), color=color)
    pred_df['Prediction'].plot(figsize=(15, 6), color='orange')
    plt.legend(loc=4)
    set_labels(ttl,x_label, y_label)

def set_labels(ttl,x_label, y_label):
    plt.title(ttl)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


