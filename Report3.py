
# coding: utf-8

# # Different types of data
# 
# + <u>Cross-sectional data</u> -
#     Cross-sectional data or cross-section of a population is obtained by taking
#     observations from multiple individuals at the same point in time.
# 
# 
# + <u>Time series data</u>-
#     A time series is made up of quantitative observations on
#     one or more measurable characteristics of an individual entity and taken at
#     multiple points in time.
#     
#     
# + <u>Panel data</u>-
#   If we observe multiple entities over multiple
#   points in time we get a panel data also known as longitudinal data.
# 

# # Dataset
# 
#                Nifty50 data from 1-04-2010 to 1-04-2018
#             
# 
# https://www.nseindia.com/products/content/equities/indices/historical_index_data.htm
# 

# + Here we have Nifty50 data for years 2010 to 2018

# In[169]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
Nifty_data=pd.read_csv("E:/summer/NIFTY50.csv",parse_dates=['Date'],index_col=['Date'])


# In[170]:


#adding a new column for years
Nifty_data['year']=Nifty_data.index.map(lambda x:x.year)


# In[171]:


#adding a new column for months
Nifty_data['month']=Nifty_data.index.map(lambda x:x.month)


# In[172]:


Nifty_data.info()


# ### We do not have Null values

# In[173]:


Nifty_data.head()


# ## Internal structures of time series
# 
#   A time series can be expressed as $x_t = f_t + s_t + c_t + e_t,$ which is a sum of the trend, seasonal, cyclical, and irregular components in that
#   order.Here, t is the time index at which observations about the series have been
# taken at t = 1,2,3 ...N successive and equally spaced points in time.
# 
#   + ***<u>General trend</u>***
#    When a time series exhibits an upward or downward movement in the long run,
# it is said to have a general trend.
# 
# + ***<u>Seasonality***</u>
#   Seasonality manifests as repetitive and period variations in a time series. In most
#   cases, exploratory data analysis reveals the presence of seasonality.
# 
# 
# + ***<u>Cyclical movements***</u>
#   Cyclical changes are movements observed after every few units of time, but they
# occur less frequently than seasonal fluctuations. Unlike seasonality, cyclical
# changes might not have a fixed period of variations.
# 
# 
# + ***<u>Unexpected variations***</u>
#   This
# fourth component reflects unexpected variations in the time series. Unexpected
# variations are stochastic and cannot be framed in a mathematical model for a
# definitive future prediction. This type of error is due to lack of information about
# explanatory variables that can model these variations or due to presence of a
# random noise.

# #### Plotting of Nifty50 data to visualize some aspects

# In[174]:


plt.plot(Nifty_data[['Open']])
plt.title('Nifty50 data for Open prices')
plt.show()
plt.plot(Nifty_data[['High']])
plt.title('Nifty50 data for High prices')
plt.show()
plt.plot(Nifty_data[['Low']])
plt.title('Nifty50 data for Low prices')
plt.show()
plt.plot(Nifty_data[['Close']])
plt.title('Nifty50 data for Close prices')
plt.show()
plt.plot(Nifty_data[['Shares Traded']])
plt.title('Nifty50 data for number of Shares Traded')
plt.show()
plt.plot(Nifty_data[['Turnover (Rs. Cr)']])
plt.title('Nifty50 data for Turnover')
plt.show()


# ##### As we can see there see there General trend is present in our data
# #### To detrend the data,we will use use Linear regression.

# In[175]:


from sklearn.linear_model import LinearRegression


# In[176]:


# fitting trend model
trend_model_Close = LinearRegression(normalize=True, fit_intercept=True)
trend_model_Close.fit(np.arange(np.array(len(Nifty_data))).reshape((-1,1)), Nifty_data['Close'])
print('Trend model coefficient={} and intercept={}'.format(trend_model_Close.coef_[0],trend_model_Close.intercept_))


# In[177]:


trend_model_Close.score(np.arange(np.array(len(Nifty_data))).reshape((-1,1)), Nifty_data['Close'])


# In[178]:


residuals_Close = np.array(Nifty_data['Close']) - trend_model_Close.predict(np.arange(np.array(len(Nifty_data))).reshape((-1,1)))
plt.figure(figsize=(5.5, 5.5))
pd.Series(data=residuals_Close, index=Nifty_data.index).plot(color='b')
plt.title('Residuals of trend model for Close prices')
plt.xlabel('Time')
plt.ylabel('Close prices')


# In[179]:


# plotting data with trend line
plt.plot(Nifty_data['Close'])
plt.plot(pd.Series(trend_model_Close.predict(np.arange(np.array(len(Nifty_data))).reshape((-1,1))),index=Nifty_data.index))
plt.xlabel('Year')
plt.ylabel('Rs.')
plt.legend()
plt.show()


# In[180]:


# adding columns to dataset i.e. Residuals_close and Quarter 
Nifty_data['Residuals_Close'] = residuals_Close
month_quarter_map = {1: 'Q1', 2: 'Q1', 3: 'Q1',
                     4: 'Q2', 5: 'Q2', 6: 'Q2',
                     7: 'Q3', 8: 'Q3', 9: 'Q3',
                     10: 'Q4', 11: 'Q4', 12: 'Q4'
                    }
Nifty_data['Quarter'] = Nifty_data['month'].map(lambda m: month_quarter_map.get(m))


# In[181]:


# Creating new subseries
seasonal_sub_series_Close = Nifty_data.groupby(by=['year', 'Quarter'])['Residuals_Close'].aggregate([np.mean, np.std])
seasonal_sub_series_Close.columns = ['Quarterly Mean Close', 'Quarterly Standard Deviation Close']


# In[182]:


#Create row indices of seasonal_sub_series_data using Year & Quarter
seasonal_sub_series_Close.reset_index(inplace=True)
seasonal_sub_series_Close.index = seasonal_sub_series_Close['year'].astype(str) + '-' + seasonal_sub_series_Close['Quarter']
seasonal_sub_series_Close.head()


# ## A practical technique of determining seasonality is through exploratory data
# ## analysis through the following plots:
# 1. Run sequence plot
# 2. Seasonal sub series plot
# 3. Multiple box plots

# ## Seasonal sub series plot
# For a known periodicity of seasonal variations, seasonal sub series redraws the
# original series over batches of successive time periods.
# A seasonal sub series reveals two properties:
# + Variations within seasons as within a batch of successive months
# + Variations between seasons as between batches of successive months

# In[183]:


# Seasonal sub series plot
plt.figure(figsize=(5.5, 5.5))
seasonal_sub_series_Close['Quarterly Mean Close'].plot(color='b')
plt.title('Quarterly Mean of Residuals')
plt.xlabel('Time')
plt.ylabel('Close prices')
plt.xticks(rotation=30)
plt.show()


# In[184]:


# Seasonal sub series plot
plt.figure(figsize=(5.5, 5.5))
seasonal_sub_series_Close['Quarterly Standard Deviation Close'].plot(color='b')
plt.title(' Quarterly Standard Deviation of Residuals')
plt.xlabel('Time')
plt.ylabel('Close prices')
plt.xticks(rotation=30)


# In[185]:


seasonal_sub_series_Close.head()


# # Multiple box plots
# A box plot displays both
# central tendency and dispersion within the seasonal data over a batch of time
# units.Besides, separation between two adjacent box plots reveal the within season variations

# In[186]:


# Multiple Boxplot(Quarterly)
import seaborn as sns
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(data=Nifty_data[['Residuals_Close','Quarter']], y=Nifty_data['Residuals_Close'], x=Nifty_data['Quarter'])
g.set_title('Quarterly Mean of Residuals')
g.set_xlabel('Time')
g.set_ylabel('Residuals_Close')


# In[187]:


# Multiple Boxplot(Yearly)
import seaborn as sns
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(data=Nifty_data[['Residuals_Close','year']], y=Nifty_data['Residuals_Close'], x=Nifty_data['year'])
g.set_title('Yearly Mean of Residuals')
g.set_xlabel('Time')
g.set_ylabel('Residuals_Close')


# In[188]:


# Multiple Boxplot(Monthly)
import seaborn as sns
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(data=Nifty_data[['Residuals_Close','month']], y=Nifty_data['Residuals_Close'], x=Nifty_data['month'])
g.set_title('Monthly Mean of Residuals')
g.set_xlabel('Time')
g.set_ylabel('Residuals_Close')


# In[189]:


seasonal_sub_series_Close.head()


# In[190]:


Nifty_data.head()


# ## Run sequence plot
# A simple run sequence plot of the original time series with time on x-axis and the
# variable on y-axis is good for indicating the following properties of the time
# series:
# + Movements in mean of the series
# + Shifts in variance
# + Presence of outliers

# In[191]:


# Run sequence plot Quarterly
plt.plot(Nifty_data[['Residuals_Close']].rolling(window=90).mean(),label='residual mean')
plt.plot(Nifty_data[['Residuals_Close']].rolling(window=90).std(),label='residual std')
plt.plot(Nifty_data[['Residuals_Close']],label='Residuals')
plt.xlabel('year')
plt.ylabel('Residuals')
plt.title('Residuals,Quarterly mean and Quarterly Std ')
plt.legend()
plt.show()


# In[192]:


# Run sequence plot Monthly
plt.plot(Nifty_data[['Residuals_Close']].rolling(window=30).mean(),label='residual mean')
plt.plot(Nifty_data[['Residuals_Close']].rolling(window=30).std(),label='residual std')
plt.plot(Nifty_data[['Residuals_Close']],label='Residuals')
plt.xlabel('year')
plt.ylabel('Residuals')
plt.title('Residuals,Monthly mean and Monthly Std ')
plt.legend()
plt.show()


# In[193]:


# Run sequence plot Yearly
plt.plot(Nifty_data[['Residuals_Close']].rolling(window=365).mean(),label='residual mean')
plt.plot(Nifty_data[['Residuals_Close']].rolling(window=365).std(),label='residual std')
plt.plot(Nifty_data[['Residuals_Close']],label='Residuals')
plt.xlabel('year')
plt.ylabel('Residuals')
plt.title('Residuals,Yearly mean and Yearly Std ')
plt.legend()
plt.show()


# ##### Ridge Regression
# 

# Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients. The ridge coefficients minimize a penalized residual sum of squares,
# 
#  Here, alpha is a complexity parameter that controls the amount of shrinkage: the larger the value of , the greater the amount of shrinkage and thus the coefficients become more robust to collinearity.

# In[194]:


# fitting the model in data using Ridge regression 
from sklearn import linear_model
ridg_reg = linear_model.Ridge (alpha = .5)
ridg_reg.fit (np.arange(np.array(len(Nifty_data))).reshape((-1,1)), Nifty_data['Close'])


# In[195]:


ridg_reg.coef_


# In[196]:


ridg_reg.intercept_ 


# ####  Setting the regularization parameter: generalized Cross-Validation
#     RidgeCV implements ridge regression with built-in cross-validation of the alpha parameter. The object works in the same   way as GridSearchCV except that it defaults to Generalized Cross-Validation (GCV)

# In[197]:


# generalized cross validation
ridg_reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
ridg_reg.fit (np.arange(np.array(len(Nifty_data))).reshape((-1,1)), Nifty_data['Close'])


# In[198]:


ridg_reg.score(np.arange(np.array(len(Nifty_data))).reshape((-1,1)), Nifty_data['Close'])


# In[199]:


ridg_reg.alpha_ 


# In[200]:


residuals_Ridge = np.array(Nifty_data['Close']) - ridg_reg.predict(np.arange(np.array(len(Nifty_data))).reshape((-1,1)))
plt.figure(figsize=(5.5, 5.5))
pd.Series(data=residuals_Ridge, index=Nifty_data.index).plot(color='b')


# In[201]:


plt.plot(Nifty_data['Close'])
plt.plot(pd.Series(ridg_reg.predict(np.arange(np.array(len(Nifty_data))).reshape((-1,1))),index=Nifty_data.index))
plt.xlabel('Year')
plt.ylabel('Rs.')
plt.legend()
plt.show()


# #### Lassao regression
#     The Lasso is a linear model that estimates sparse coefficients. It is useful in some contexts due to its tendency to      prefer solutions with fewer parameter values, effectively reducing the number of variables upon which the given solution is  dependent. For this reason, the Lasso and its variants are fundamental to the field of compressed sensing. Under     certain  conditions, it can recover the exact set of non-zero weights
#     
#     
#     

# In[202]:


#fitting lassao Regression
lass_reg = linear_model.Lasso(alpha = 0.1)
lass_reg.fit(np.arange(np.array(len(Nifty_data))).reshape((-1,1)), Nifty_data['Close'])


# In[203]:


residuals_lassao = np.array(Nifty_data['Close']) - ridg_reg.predict(np.arange(np.array(len(Nifty_data))).reshape((-1,1)))
plt.figure(figsize=(5.5, 5.5))
pd.Series(data=residuals_Ridge, index=Nifty_data.index).plot(color='b')


# In[204]:


lass_reg.score(np.arange(np.array(len(Nifty_data))).reshape((-1,1)), Nifty_data['Close']) 


# In[205]:


lass_reg.predict(np.arange(np.array(len(Nifty_data))).reshape((-1,1)))


# In[206]:


plt.plot(Nifty_data['Close'])
plt.plot(pd.Series(lass_reg.predict(np.arange(np.array(len(Nifty_data))).reshape((-1,1))),index=Nifty_data.index))
plt.xlabel('Year')
plt.ylabel('Rs.')
plt.legend()
plt.show()


# # baysienRidge

# In[207]:


# fittng model using baysienRidge Regression
bay_reg = linear_model.BayesianRidge()
bay_reg.fit(np.arange(np.array(len(Nifty_data))).reshape((-1,1)), Nifty_data['Close'])


# In[208]:


plt.plot(Nifty_data['Close'])
plt.plot(pd.Series(bay_reg.predict(np.arange(np.array(len(Nifty_data))).reshape((-1,1))),index=Nifty_data.index))
plt.xlabel('Year')
plt.ylabel('Rs.')
plt.legend()
plt.show()


# In[209]:


bay_reg.score(np.arange(np.array(len(Nifty_data))).reshape((-1,1)), Nifty_data['Close'])


# In[210]:


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=5)
    rolstd = pd.rolling_std(timeseries, window=5)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='grey',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    if(dftest[1]>0.05):
        print("Time Series is not stationary")
    else:
        print("Time series is stationary")
     


# In[211]:


plt.plot(residuals_Close)


# In[212]:


#Applying dickey fuller test on residuals
test_stationarity(Nifty_data['Residuals_Close'])


# In[213]:


# calculating Moving -average
Moving_avg_res=Nifty_data['Residuals_Close'].rolling(window=5,center=True).mean()
# the center is true to calculate symmetrical moving average


# In[214]:


# apply differencing using moving average
stationary_close=pd.Series(Nifty_data['Residuals_Close'],index=Nifty_data.index)-Moving_avg_res


# In[215]:


stationary_close=stationary_close.dropna()


# In[216]:


test_stationarity(stationary_close)


# **Time series is become stationary**

# In[217]:


from statsmodels.tsa import seasonal
decompose_model5 = seasonal.seasonal_decompose(Nifty_data['Close'],freq=5)


# In[218]:


decompose_model5.plot()


# In[219]:


decompose_model5.resid


# In[220]:



decompose_model21 = seasonal.seasonal_decompose(Nifty_data['Close'],freq=21)
decompose_model21.plot()


# In[221]:



decompose_model63 = seasonal.seasonal_decompose(Nifty_data['Close'],freq=63)
decompose_model63.plot()


# In[222]:



decompose_model252 = seasonal.seasonal_decompose(Nifty_data['Close'],freq=252)
decompose_model252.plot()


# In[223]:


def single_smoothing(x, alpha):
    F = [x[0]] # first value is same as series
    for t in range(1, len(x)):
        F.append(alpha * x[t] + (1 - alpha) * F[t-1])
    return F
Nifty_data['Single_Exponential_Forecast'] = single_smoothing(Nifty_data['Close'], 0.4)


# In[224]:


Nifty_data.head()


# In[225]:


Nifty_data['Close'].plot(color='b')
Nifty_data['Single_Exponential_Forecast'].plot(color='r')


# In[226]:


def double_exp_smoothing(x, alpha, beta):
    yhat = [x[0]] # first value is same as series
    for t in range(1, len(x)):
        if t==1:
            F, T= x[0], x[1] - x[0]
        F_n_1, F = F, alpha*x[t] + (1-alpha)*(F+T)
        T=beta*(F-F_n_1)+(1-beta)*T
        yhat.append(F+T)
    return yhat

Nifty_data['DEF'] = double_exp_smoothing(Nifty_data['Close'], 0.4, 0.7)


# In[227]:


Nifty_data['Close'].plot(color='b')
Nifty_data['DEF'].plot(color='r')


# In[228]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
import statsmodels.tsa.api as smtsa


# In[245]:


# Function to plot signal, ACF and PACF
def plotds(xt, nlag=21, fig_size=(12, 10)):
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


# In[246]:


# Number of samples
n = len(Nifty_data['Close'])


# In[247]:


ar = np.r_[Nifty_data['Close'], 0.6]
ma = np.r_[Nifty_data['Close'], 0]
ar1_data = smtsa.arma_generate_sample(ar=ar, ma=ma, nsample=n) 
plotds(ar1_data)


# In[248]:


# Generate AR(2) dataset
ar = np.r_[Nifty_data['Close'], 0.6, 0.7]
ma = np.r_[Nifty_data['Close'], 0]
ar2_data = smtsa.arma_generate_sample(ar=ar, ma=ma, nsample=n) 
plotds(ar2_data)


# In[249]:


ar = np.r_[Nifty_data['Close'], 0.6, 0.7, 0.5]
ma = np.r_[Nifty_data['Close'], 0]
ar3_data = smtsa.arma_generate_sample(ar=ar, ma=ma, nsample=n) 
plotds(ar3_data)


# In[250]:


# Build AR(1) model
ar1model = smtsa.ARMA(ar1_data.tolist(), order=(1, 0))
ar1=ar1model.fit(maxlag=21, method='mle', trend='nc')
ar1.summary()


# In[251]:


ar3 = smtsa.ARMA(ar3_data.tolist(), order=(3, 0)).fit(maxlag=21, method='mle', trend='nc')
ar3.summary()

