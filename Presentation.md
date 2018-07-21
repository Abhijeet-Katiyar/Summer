<center><u>**Introduction**</u></center>
================================

##  <u>Different Types of data </u>-

+ <u>**Cross-sectional data**</u> : Cross sectional data can obtained by taking multiple observation from multiple individuals at same point in time.
+ <u>**Timeseries data**</u> : Timeseries data can obtained by taking multiple observations from same source at different points of time.
+ <u>**Panel data**</u> : Panel data is collection of multiple observations over multiple points in time. It is combination of cross-sectional data and Time-series data. 

The **Nifty50** data that used is Time series data from **APR-2010** to **MAR-2018**.





<center><u>Internal structure of time series</u></center>
====

A Time series is a combination of General trend, Seasonality, Cyclic movements and Unexpected variations.

+ <u>**General Trend**</u> : When there is Upward or downward movement present in data in a long run, is Known as general trend.


+ <u>**Seasonality**</u> : If repetitive patterns present in data which occurs over known periods of time are known as seasonality.


+ <u>**Cyclical movements**</u> : If there are movements observes after every few units of time and do not have fixed periods of variations are known as cyclic movements.


+ <u>**Unexpected variations**</u> : Occurance of sudden changes in time series which are rarely repeted. This component also known as residuals.





<center><u>Stationary time series</u><center>
===

A timeseries is known as stationary when it is free from Trend and seasonility. Its statistical properties like mean, variance, autocorrelation etc are constant over time.
+ <u>**check stationarity of timeseries**</u> : To check stationarity of timeseries we can-
    1. **Plot Rolling statistics timeseries**
    2. **Apply Augmented Dickey Fuller test**

+ By plotting Rolling statistics we can easily identify trend component.
+ Augmented Dickey fuller test is statistical test to check the stationarity of timeseries. It uses null hypothesis testing where H<sub>0</sub> rejected if p-value is greater than 0.05.


<center><u>Methods to detrending data</center></u>
===

1. Differencing
2. Regression
3. Statistical function

+ **<u>Differencing</u>** : Differencing is processs of taking difference original timeseries with itself by lag.
	example of time series with lag 1 -
	> &Delta;x<sub>t</sub> = x<sub>t</sub> - x<sub>t-1</sub>

	Where, &Delta;x<sub>t</sub> is stationary time series.
	x<sub>t</sub> is original time series.
	x<sub>t-1</sub> is time series with lag 1.

+ **<u>Regression</u>** : Regression is useful to find trend line and to remove trend component, take difference between original time series and trend line. after removing trend we will get Residuals.
+  **<u>Statistical function</u>** : In python a function named `seasonal_decompose` is present in library `statsmodels.tsa.seasonal` which separate Observed data(i.e. original data), trend component, seasonal component and residuals. 

<center><u>Forecasting</u></center>
===

There are many Statistical models for timeseries forecasting. Among them ARIMA is widely used model which is combination of Autoregressive,Integration(differencing) and Moving average models.

+ **<u>Autoregression</u>** : This model gives output which depends on its own previous values.
+ **<u>Differencing</u>** : Integration or differencing makes series sattionary.
+ **<u>Moving</u>** : This model analyze data points by creating series of averages of subsets of data.
