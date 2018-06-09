# Dataset

               Nifty50 data from 1-04-2010 to 1-04-2018
            

https://www.nseindia.com/products/content/equities/indices/historical_index_data.htm

# Different types of data

+ <u>Cross-sectional data</u>

  Cross-sectional data or cross-section of a population is obtained by taking
  observations from multiple individuals at the same point in time.

+ <u>Time series data</u>
    
    a time series is made up of quantitative observations on
    one or more measurable characteristics of an individual entity and taken at
    multiple points in time.
+ Panel data

  If we observe multiple entities over multiple
  points in time we get a panel data also known as longitudinal data.


# Internal structures of time series

  A time series can be expressed as $x_t = f_t + s_t + c_t + e_t,$ which is a sum of the trend, seasonal, cyclical, and irregular components in that
  order.Here, t is the time index at which observations about the series have been
taken at t = 1,2,3 ...N successive and equally spaced points in time.

  + ***<u>General trend</u>***
   When a time series exhibits an upward or downward movement in the long run,
it is said to have a general trend.

+ ***<u>Seasonality***</u>

  Seasonality manifests as repetitive and period variations in a time series. In most
  cases, exploratory data analysis reveals the presence of seasonality.
+ ***<u>Cyclical movements***</u>

  Cyclical changes are movements observed after every few units of time, but they
occur less frequently than seasonal fluctuations. Unlike seasonality, cyclical
changes might not have a fixed period of variations.
+ ***<u>Unexpected variations***</u>

  This
fourth component reflects unexpected variations in the time series. Unexpected
variations are stochastic and cannot be framed in a mathematical model for a
definitive future prediction. This type of error is due to lack of information about
explanatory variables that can model these variations or due to presence of a
random noise.





# Stationary time series

<p>In time series analysis, this assumption is known as
stationarity, which requires that the internal structures of the series do not change over time.A stationary time series have constant mean and constant variance over time.</p>

## Properties of Stationary time series

 A stationary time series should not have

1. <u>**Trend**</u> meaning that, on average, the measurements tend to  increase (or decrease) over time
2.  <u>**Seasonality**</u>  meaning that there is a regularly repeating pattern of highs and lows related to calendar time such as seasons, quarters, months, days of the week, and so on

# How to check whether the time series is stationary or not
The statistical tests for objectively determining whether a time series is stationary or not , are known as unit root tests. There are several such
tests of which we discuss the _**ADF(Augmented Dickey-Fuller test)**_ test is one of the unit root tests, that is most
commonly used for verifying non-stationarity in the original time series.



# Augmented Dickey-Fuller test

+ Given an observed time series _**$Y_1,Y_2,Y_3,....,Y_N$**_  ADF Time Series Dickey and Fuller consider three differential-form autoregressive equations to detect the presence of a unit root:

     
> $$\triangle Y_t=\gamma Y_t-_1+\sum_{j=1}^{p} (\delta_j\triangle Y_t-_j)+e_t$$

Where ,
- t is the time index,
- α is an intercept constant called a drift,
- β is the coefficient on a time trend,
- γ is the coefficient presenting process root, i.e. the focus of testing,
- p is the lag order of the first-differences autoregressive process,
- et is an independent identically distributes residual term.


ADF tests the null hypothesis that a unit root is present in time series sample. ADF statistic is a negative number and more negative it is the stronger the rejection of the hypothesis that there is a unit root.

#### Code used in Python

```python
    result = stattools.adfuller(test_series,autolag='AIC')

    print('ADF Statistic: %f' % result[0])

    print('p-value: %f' % result[1])

    print('Critical Values:')

    for key, value in result[4].items():

        print('\t%s: %.3f' % (key, value))

```

+ *Null Hypotehsis (H0):* If accepted, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.

+ *Alternate Hypothesis (H1):* The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary.

+ *p-value > 0.05:* Accept H0, the data has a unit root and is non-stationary

+ *p-value ≤ 0.05:* Reject H0. the data does not have a unit root and is stationary