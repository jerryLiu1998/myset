import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

'''
1-Load Data
Most sm.datasets hold convenient representations of the data in the attributes endog and exog.
If the dataset does not have a clear interpretation of what should be an endog and exog,
then you can always access the data or raw_data attributes.
https://docs.scipy.org/doc/numpy/reference/generated/numpy.recarray.html
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html
# resample('MS') groups the data in buckets by start of the month,
# after that we got only one value for each month that is the mean of the month
# fillna() fills NA/NaN values with specified method
# 'bfill' method use Next valid observation to fill gap
# If the value for June is NaN while that for July is not, we adopt the same value
# as in July for that in June
'''

data = sm.datasets.co2.load_pandas()
y = data.data  # DataFrame with attributes y.columns & y.index (DatetimeIndex)
print(y)
names = data.names  # tuple
raw = data.raw_data  # float64 np.recarray

y = y['co2'].resample('MS').mean()
print(y)

y = y.fillna(y.bfill()) # y = y.fillna(method='bfill')
print(y)

y.plot(figsize=(15,6))
plt.show()

'''
2-ARIMA Parameter Seletion
ARIMA(p,d,q)(P,D,Q)s
non-seasonal parameters: p,d,q
seasonal parameters: P,D,Q
s: period of time series, s=12 for annual period
Grid Search to find the best combination of parameters
Use AIC value to judge models, choose the parameter combination whose
AIC value is the smallest
https://docs.python.org/3/library/itertools.html
http://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
'''

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


'''
3-Optimal Model Analysis
Use the best parameter combination to construct ARIMA model
Here we use ARIMA(1,1,1)(1,1,1)12
the output coef represents the importance of each feature
mod.fit() returnType: MLEResults
http://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.MLEResults.html#statsmodels.tsa.statespace.mlemodel.MLEResults
Use plot_diagnostics() to check if parameters are against the model hypothesis
model residuals must not be correlated
'''

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()

'''
4-Make Predictions
get_prediction(..., dynamic=False)
Prediction of each point will use all historic observations prior to it
http://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.mlemodel.MLEResults.get_prediction.html#statsmodels.regression.recursive_ls.MLEResults.get_prediction
http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.plot.html
https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.fill_between.html
'''

pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
pred_ci = pred.conf_int() # return the confidence interval of fitted parameters

# plot real values and predicted values
# pred.predicted_mean is a pandas series
ax = y['1990':].plot(label='observed')  # ax is a matplotlib.axes.Axes
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

# fill_between(x,y,z) fills the area between two horizontal curves defined by (x,y)
# and (x,z). And alpha refers to the alpha transparencies
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()

plt.show()

# Evaluation of model
y_forecasted = pred.predicted_mean
y_truth = y['1998-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

'''
5-Dynamic Prediction
get_prediction(..., dynamic=True)
Prediction of each point will use all historic observations prior to 'start' and
all predicted values prior to the point to predict
'''

pred_dynamic = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

ax = y['1990':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1998-01-01'), y.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')

plt.legend()
plt.show()

# Extract the predicted and true values of our time-series
y_forecasted = pred_dynamic.predicted_mean
y_truth = y['1998-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

'''
6-Visualize Prediction
In-sample forecast: forecasting for an observation that was part of the data sample;
Out-of-sample forecast: forecasting for an observation that was not part of the data sample.
'''

# Get forecast 500 steps ahead in future
# 'steps': If an integer, the number of steps to forecast from the end of the sample.
pred_uc = results.get_forecast(steps=500)  # retun out-of-sample forecast

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')

plt.legend()
plt.show()
