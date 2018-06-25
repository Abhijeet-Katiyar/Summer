# Dataset

  >             Nifty50 data from Apr-2010 to Mar-2018


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

  A time series can be expressed as $x<sub>t</sub> = f<sub>t</sub> + s<sub>t</sub> + c<sub>t</sub> + e<sub>t</sub>, $ which is a sum of the trend, seasonal, cyclical, and irregular components in that
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

+ Given an observed time series  _**Y<sub>1</sub>,Y<sub>2</sub>,Y<sub>3</sub>,....,Y<sub>N</sub>**_   ADF Time Series Dickey and Fuller consider three differential-form autoregressive equations to detect the presence of a unit root:


> ΔY<sub>t</sub> = &gamma;Y<sub>t-1</sub> + &Sigma;(&delta;<sub>j</sub>&Delta;Y<sub>t-j</sub>) + e<sub>t</sub>


Where ,
- t is the time index,
- α is an intercept constant called a drift,
- β is the coefficient on a time trend,
- γ is the coefficient presenting process root, i.e. the focus of testing,
- p is the lag order of the first-differences autoregressive process,
- et is an independent identically distributes residual term.


ADF tests the null hypothesis that a unit root is present in time series sample. ADF statistic is a negative number and more negative it is the stronger the rejection of the hypothesis that there is a unit root.

#### <u>Code used in Python</u>

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


# Gain stationarity in data

## <u>Detrending the data:</u>

A general trend is commonly modeled by setting up the time series as a
regression against time and other known factors as explanatory variables. The
regression or trend line can then be used as a prediction of the long run
movement of the time series. Residuals left by the trend line is further analyzed
for other interesting properties such as seasonality, cyclical behavior, and
irregular variations.

<u>to detrending the data,python code used:</u>
```Python
from sklearn.linear_model import LinearRegression
# fitting trend model
trend_model_Close = LinearRegression(normalize=True, fit_intercept=True)
trend_model_Close.fit(np.arange(np.array(len(Nifty_data))).reshape((-1,1)), Nifty_data['Close'])
print('Trend model coefficient={} and intercept={}'.format(trend_model_Close.coef_[0],trend_model_Close.intercept_))
trend_model_Close.score(np.arange(np.array(len(Nifty_data))).reshape((-1,1)), Nifty_data['Close'])

residuals_Close = np.array(Nifty_data['Close']) - trend_model_Close.predict(np.arange(np.array(len(Nifty_data))).reshape((-1,1)))
plt.figure(figsize=(5.5, 5.5))
pd.Series(data=residuals_Close, index=Nifty_data.index).plot(color='b')
plt.title('Residuals of trend model for Close prices')
plt.xlabel('Time')
plt.ylabel('Close prices')
```


*Here Linear LinearRegression is used to de trending the Dataset*

LinearRegression fits a linear model with coefficients  to minimize the residual sum of squares between the observed responses in the dataset, and the responses predicted by the linear approximation.




  A residual is the difference between what is plotted at a specific point, and what the regression equation predicts "should be plotted" at this specific point. If the scatter plot and the regression equation "agree" on a y-value (no difference), the residual will be zero.


*Residuals are used as detrended data*

   **Residual = Observed y-value - Predicted y-value**

  A residual is the difference between the observed y-value (from scatter plot) and the predicted y-value (from regression equation line).    
It is the vertical distance from the actual plotted point to the point on the regression line.
You can think of a residual as how far the data "fall" from the regression line
(sometimes referred to as "observed error").



## A practical technique of determining seasonality is through exploratory data
## analysis through the following plots:
1. Run sequence plot
2. Seasonal sub series plot
3. Multiple box plots

## Run sequence plot
A simple run sequence plot of the original time series with time on x-axis and the
variable on y-axis is good for indicating the following properties of the time
series:
+ Movements in mean of the series
+ Shifts in variance
+ Presence of outliers


## Seasonal sub series plot
For a known periodicity of seasonal variations, seasonal sub series redraws the
original series over batches of successive time periods.
A seasonal sub series reveals two properties:
+ Variations within seasons as within a batch of successive months
+ Variations between seasons as between batches of successive months

## Multiple box plots
A box plot displays both
central tendency and dispersion within the seasonal data over a batch of time
units.Besides, separation between two adjacent box plots reveal the within season variations



# Auto-Regressive Models

Another very famous approach to regress on time series data is to regress it with
its lag term. This genre of models is referred to as auto-regressive models (AR
models). The AR models are very good in capturing trends as the next time
values are predicted based on the prior time values.Thus AR models are very
useful in situations where the next forecasted value is a function of the previous
time period

The auto-regressive model is defined as AR(p), where p refers to the order of the
AR component.

The first-order AR model is denoted by AR(1):
> x<sub>t</sub> = ø∈<sub>t-1</sub> + ∈<sub>t</sub>

The second-order AR model is denoted by AR(2):
> x<sub>t</sub> = ø<sub>1</sub>∈<sub>t-1</sub> + ø<sub>2</sub>∈<sub>t-2</sub> + ∈<sub>t</sub>


The p<sup>th</sup> order AR model is denoted by AR(p):
> x<sub>t</sub> = ø<sub>1</sub>∈<sub>t-1</sub> + ø<sub>2</sub>∈<sub>t-2</sub> + ... + ø<sub>p</sub>∈<sub>t-p</sub> + ∈<sub>t</sub>

Here, ø is the model coefficient, ∈<sub>t</sub> ∼ N (0, σ<sup>2</sup>) is an error in time t, and p is the
order of the AR model.

# Moving average models
The moving average models use dependency between residual errors to forecast
values in the next time period. The model helps you adjust for any unpredictable
events such as catastrophic events leading to a share market crash leading to
share prices falling, which will happen over time and is captured as a moving
average process.

The first-order moving average denoted by MA(1) is as follows:  
> x<sub>t</sub> = α - θ<sub>1</sub>Є<sub>t-1</sub> + Є<sub>t</sub>

The second-order moving average denoted by MA(2) is as follows:  
> x<sub>t</sub> = α - θ<sub>1</sub>Є<sub>t-1</sub> - θ<sub>2</sub>Є<sub>t-2</sub>+ Є<sub>t</sub>

The qth order moving average denoted by MA(q) is as follows:
> x<sub>t</sub> = α - θ<sub>1</sub>Є<sub>t-1</sub> - θ<sub>2</sub>Є<sub>t-2</sub> - ... - θ<sub>q</sub>Є<sub>t-q</sub>+ Є<sub>t</sub>

Here, Є<sub>t</sub> is the identically independently-distributed error at time t and follows
normal distribution N(0,σ<sup>2</sup><sub>
Є</sub>) with zero mean and σ<sup>2</sup><sub>
Є</sub> variance. The Є<sub>t</sub>
component represents error in time t and the α and Є notations represent mean
intercept and error coefficients, respectively. The moving average time series
model with q<sup>th</sup> order is represented as MA(q).

# Building datasets with ARMA

The preceding two sections describe the auto-regressive model AR(p), which
regresses on its own lagged terms and moving average model MA(q) builds a
function of error terms of the past. The AR(p) models tend to capture the mean
reversion effect whereas MA(q) models tend to capture the shock effect in error
,which are not normal or unpredicted events.
Thus, the ARMA model combines
the power of AR and MA components together. An ARMA(p, q) time series
forecasting model incorporates the pth order AR and qth order MA model,
respectively.

The ARMA (1, 1) model is represented as follows:
> x<sub>t</sub> = &alpha; + &straightphi;<sub>1</sub>x<sub>t-1</sub> - &theta;<sub>1</sub>&epsilon;<sub>t-1</sub> + &epsilon;<sub>t</sub>

The ARMA(1, 2) model is denoted as follows:
> x<sub>t</sub> = &alpha; + &straightphi;<sub>1</sub>x<sub>t-1</sub> - &theta;<sub>1</sub>&epsilon;<sub>t-1</sub> - &theta;<sub>2</sub>&epsilon;<sub>t-2</sub> + &epsilon;<sub>t</sub>

The ARMA(p, q) model is denoted as follows:
> x<sub>t</sub> = &alpha; + &straightphi;<sub>1</sub>x<sub>t-1</sub> + &straightphi;<sub>2</sub>x<sub>t-2</sub> + ... +  &straightphi;<sub>p</sub>x<sub>t-p</sub> - &theta;<sub>1</sub>&epsilon;<sub>t-1</sub> - &theta;<sub>2</sub>&epsilon;<sub>t-2</sub> - ... - &theta;<sub>q</sub>&epsilon;<sub>t-q</sub> + &epsilon;<sub>t</sub>

Here, Φ and θ represent AR and MA coefficients. The α and εt captures the
intercept and error at time t. The form gets very complicated as p and q increase;
thus, lag operators are utilized for a concise representation of ARMA models.

There are multiple
scenarios to select p and q; some of the thumb rules that can be used to
determine the order of ARMA components are as follows:

+ Autocorrelation is exponentially decreasing and PACF has significant
correlation at lag 1, then use the p parameter  

+ Autocorrelation is forming a sine-wave and PACF has significant
correlation at lags 1 and 2, then use second-order value for p
+ Autocorrelation has significant autocorrelation and PACF has exponential
decay, then moving average is present and the q parameter needs to be set
up
+ Autocorrelation shows significant serial correlation and the PACF shows
sine-wave pattern, then set up a moving average q parameter

One of the major limitations of these models are that they ignore the volatility
factor making the signal non-stationary. The AR modeling is under consideration
process is stationary, that is, error term is IID and follows normal distribution εt
∼ N(0,σ<sup>2</sup><sub>
ε</sub>) and |Φ|<1. The |Φ|<1 condition makes the time series a finite time
series as the effect of more recent observations in time series would be higher as
compared to prior observations. The series that do not satisfy these assumptions
fall into non-stationary series.


# ARIMA

ARIMA, also known as the Box-Jenkins model, is a generalization of the ARMA
model by including integrated components. The integrated components are
useful when data has non-stationarity, and the integrated part of ARIMA helps in
reducing the non-stationarity. The ARIMA applies differencing on time series
one or more times to remove non-stationarity effect. The ARIMA(p, d, q)
represent the order for AR, MA, and differencing components. The major
difference between ARMA and ARIMA models is the d component, which
updates the series on which forecasting model is built. The d component aims to
de-trend the signal to make it stationary and ARMA model can be applied to the
de-trended dataset.For different values of d, the series response changes as
follows:
+ For d=0: xt =xt
+ For d=1: xt =xt - xt-1
+ For d=2: xt =(xt - xt-1) - (xt-1 - xt-2) = xt- 2xt-1- xt-2


As can be seen from the preceding lines, the second difference is not two periods
ago, rather it is the difference of the first different, that is, d=1. Let's say that
represents the differenced response and so ARIMA forecasting can be written as
follows:
> x<sub>t</sub> = &Phi;<sub>1</sub>x<sub>t-1</sub> + &Phi;<sub>2</sub>x<sub>t-2</sub> + ..... + &Phi;<sub>p</sub>x<sub>t-q</sub> + &theta;<sub>1</sub>&epsilon;<sub>t-1</sub> + &theta;<sub>2</sub>&epsilon;<sub>t-2</sub> + ..... + &theta;<sub>q</sub>&epsilon;<sub>t-q</sub> + &epsilon;<sub>t</sub>
