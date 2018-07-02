
Pyramid

Pyramid brings R's beloved auto.arima to Python, making an even stronger case for why you don't need R for data science. It does so not by calling R under the hood, but by wrapping statsmodels' well-tested ARIMA and SARIMAX estimators in a single, easy-to-use scikit-learn-esque estimator.



https://github.com/tgsmith61591/pyramid

```
    if xindex==0:
        stepwise_fit = auto_arima(kpi, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                              start_P=0, seasonal=True, d=1, D=1, trace=True,
                              error_action='ignore',  # don't want to know if an order does not work
                              suppress_warnings=True,  # don't want convergence warnings
                              stepwise=True)  # set to stepwise
    else:
        stepwise_fit.fit(kpi)
```
