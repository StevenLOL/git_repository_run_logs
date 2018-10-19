Multicore-TSNE

https://github.com/DmitryUlyanov/Multicore-TSNE


```
from MulticoreTSNE import MulticoreTSNE
tsneMC=MulticoreTSNE(n_components=2,perplexity=50,n_jobs=-1)
tsneMCData=tsneMC.fit_transform(trainx)

```


```
from ggplot import *
print(tsneMCData.shape)
tempDF=pd.DataFrame()
tempDF['x-tsne']=tsneMCData[:,0]
tempDF['y-tsne']=tsneMCData[:,1]
plot.figure(figsize=(15,8))
#Counter({'MTRS-1E21-01   ': 83515, 'MXPD-243S      ': 10}) 
#plot.scatter(tsneMCData[:,0],tsneMCData[:,1],c='r')
plot.scatter(tsneMCData[trainPNs=='MTRS-1E21-01   ',0],tsneMCData[trainPNs=='MTRS-1E21-01   ',1],c='r',s=1)
plot.scatter(tsneMCData[trainPNs=='MXPD-243S      ',0],tsneMCData[trainPNs=='MXPD-243S      ',1],c='b',s=1)
```
