
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[2]:


Nifty_data=pd.read_csv("E:/summer/NIFTY50.csv",parse_dates=['Date'],index_col=['Date'])


# In[3]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 


# In[4]:


train=Nifty_data['Close'].iloc[:1750]
test=Nifty_data['Close'].iloc[1751:]


# In[5]:


train.tail()


# In[6]:


test.head()


# In[7]:


from statsmodels.tsa.arima_model import ARIMA


# In[8]:


def plotds(xt, nlag=30, fig_size=(12, 10)):
    if not isinstance(xt, pd.Series):
         xt = pd.Series(xt)
    plt.figure(figsize=fig_size)
    layout = (2, 2)
    
    # Assign axes
    ax_xt = plt.subplot2grid(layout, (0, 0), colspan=2)
    ax_acf= plt.subplot2grid(layout, (1, 0))
    ax_pacf = plt.subplot2grid(layout, (1, 1))
    
    # Plot graphs
    xt.plot(ax=ax_xt)
    ax_xt.set_title('Time Series')
    plot_acf(xt, lags=50, ax=ax_acf)
    plot_pacf(xt, lags=50, ax=ax_pacf)
    plt.tight_layout()
    return None


# In[9]:


# plotting data
plotds(Nifty_data['Close'], nlag=50)


# In[10]:


#plotting QQ plot and probability plot
sm.qqplot(Nifty_data['Close'], line='s')


# In[25]:


# Optimize ARIMA parameters
aicVal=[]
for d in range(0,3):
    for ari in range(0,3):
        for maj in range(0,3):
            try:
                arima_obj1 = ARIMA(train.tolist(), order=(ari,d,maj))
                arima_obj1_fit=arima_obj1.fit()
                aicVal.append([ari, d, maj, arima_obj1_fit.aic])
            except:
                pass


# In[26]:


aicVal


# In[27]:


pred=np.append([0,0],arima_obj1_fit.fittedvalues.tolist())


# In[28]:


import sklearn


# In[29]:


sklearn.metrics.r2_score(arima_obj1_fit,pred)

