
##data 
A pandas Series with two fields(timeStamp, values). The timeStamp is the index.  

## Prophet
```
finalP=[]
trueP=[]
from fbprophet import Prophet
for xindex,x in enumerate(x2[500:].values):    
    xtrain=pd.concat([x1,x2[:500+xindex]])
    xtrain.columns=['ds','y']
    print(xindex,x,xtrain.shape)
    m = Prophet()
    m.fit(xtrain)
    future=m.make_future_dataframe(periods=1,freq='10S',include_history=False)
    forecast = m.predict(future)
    finalP.append(forecast['yhat'])
    trueP.append(x[1])
plt.plot(finalP,'r')
plt.plot(trueP,'b')
print(mean_squared_error(np.array(trueP),np.array(finalP)))
print(mean_absolute_error(np.array(trueP),np.array(finalP)))
print(NRMSE(np.array(trueP),np.array(finalP)))
```



```
finalP=[]
trueP=[]


t1=time.time()
from fbprophet import Prophet
for xindex,x in enumerate(x2[500:].values):    
    xtrain=pd.concat([x1,x2[:500+xindex]])
    xtrain.columns=['ds','y']

    xtrain=xtrain.set_index('ds')
    kpi=xtrain['y'].astype(np.float64)
    
    model = ARIMA(kpi, order=(2,1,1)).fit()
    pred24,_c2,_c1 = model.forecast(1)

    finalP.append(pred24)
    trueP.append(x[1])
    print(xindex,x,xtrain.shape,time.time()-t1)
    t1=time.time()
    
    
plt.plot(finalP,'r')
plt.plot(trueP,'b')
print(mean_squared_error(np.array(trueP),np.array(finalP)))
print(mean_absolute_error(np.array(trueP),np.array(finalP)))
print(NRMSE(np.array(trueP),np.array(finalP)))
```



```
#Normalised root meansquare error
def NRMSE(ref,predict):
    #rmse=np.sqrt( np.sum(np.square(ref-predict))/len(ref))
    rmse=np.sqrt(mean_squared_error(ref,predict))
    return rmse/np.absolute(np.max(ref)-np.min(ref))
    
    
    

```




