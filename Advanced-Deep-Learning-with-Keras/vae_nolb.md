

```python
'''Example of CVAE on MNIST dataset using CNN

This VAE has a modular design. The encoder, decoder and vae
are 3 models that share weights. After training vae,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a gaussian dist with mean=0 and std=1.

[1] Sohn, Kihyuk, Honglak Lee, and Xinchen Yan.
"Learning structured output representation using
deep conditional generative models."
Advances in Neural Information Processing Systems. 2015.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd

from pandas.tseries.offsets import *
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
import glob
from locale import atof
import numpy as np
%matplotlib inline
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from sklearn.model_selection import LeaveOneOut,StratifiedKFold,KFold
from sklearn.svm import LinearSVR,LinearSVC, SVR ,SVC  #82
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,RandomForestClassifier
from sklearn.metrics import mean_absolute_error,classification_report,confusion_matrix,mean_squared_error
from sklearn.linear_model import  LinearRegression,LogisticRegression,HuberRegressor   #82
from sklearn.neural_network import MLPRegressor, MLPClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import tqdm
from sklearn.metrics import classification_report
from sklearn.ensemble import BaggingRegressor
from lightgbm import LGBMRegressor

import tensorflow as tf
import pandas as pd
import keras.backend.tensorflow_backend as K
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential,Model
from keras.optimizers import Adam,RMSprop
from keras.activations import tanh,relu
#from keras.utils import multi_gpu_model
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU
from keras.layers import Dense, Dropout, Activation, Flatten,LSTM,GRU,Input,InputLayer,Activation, Input,Conv1D,MaxPooling1D,GlobalAveragePooling1D
from keras.layers import Convolution2D, MaxPooling2D,TimeDistributed,Convolution1D,MaxPooling1D,concatenate, Average,BatchNormalization,GlobalMaxPool1D
from keras.utils import np_utils
from keras import losses
#from keras_tqdm import TQDMNotebookCallback
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import sklearn
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error,classification_report
print (sklearn.__version__)
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import BaggingRegressor
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
plt.rcParams['font.sans-serif']=['SimHei']
__author__ = 'Marco De Nadai'
__license__ = "MIT"
import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import datetime
#import matplotlib.pyplot as plt
import pandas as pd
import glob
import tqdm
import datetime
import keras
import numpy as np
#test getWindowedValue
import numpy as np
from numpy.lib.stride_tricks import as_strided
from keras.models import Sequential
from keras.layers.convolutional import Conv3D,Conv2D,MaxPooling1D,MaxPooling2D,MaxPooling3D,Conv1D
from keras.layers import Lambda,Multiply ,TimeDistributed
from keras.layers import GlobalAveragePooling1D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.recurrent import LSTM,GRU
from keras.layers.wrappers import TimeDistributed
from keras import initializers
from keras.engine import InputSpec, Layer
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
%matplotlib inline
import pickle
from IPython.display import SVG
from matplotlib import pyplot
import keras
from keras.utils.vis_utils import model_to_dot


#Normalised root meansquare error
def NRMSE(ref,predict):
    #rmse=np.sqrt( np.sum(np.square(ref-predict))/len(ref))
    rmse=np.sqrt(mean_squared_error(ref,predict))
    return rmse/np.absolute(np.max(ref)-np.min(ref))


def mean_absolute_percentage_error(y_true, y_pred):    

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    y_true[y_true==0]=0.000000001
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import warnings
warnings.filterwarnings('ignore')

from keras.layers import Embedding
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
import keras
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

%matplotlib inline
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Implements reparameterization trick by sampling
    from a gaussian with zero mean and std=1.

    Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    Returns:
        sampled latent vector (tensor)
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 y_label,
                 batch_size=128,
                 model_name="cvae_mnist"):
    """Plots 2-dim mean values of Q(z|X) using labels as color gradient
        then, plot MNIST digits as function of 2-dim latent vector

    Arguments:
        models (list): encoder and decoder models
        data (list): test data and label
        y_label (array): one-hot vector of which digit to plot
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "cvae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict([x_test, to_categorical(y_test)],
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "%05d.png" % np.argmax(y_label))
    # display a 10x10 2D manifold of the digit (y_label)
    n = 10
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict([z_sample, y_label])
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
```

    0.19.0


    /usr/local/lib/python3.5/dist-packages/sklearn/grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)



```python
# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# compute the number of labels
#um_labels = len(np.unique(y_train))


```


```python
# network parameters
featureDIM=122
input_shape = (featureDIM,1)
#abel_shape = (num_labels, )
batch_size = 128
#ernel_size = 3
#ilters = 16
latent_dim = 14
epochs = 20
```


```python
plt.imshow(trainx[:10].squeeze())
print(trainx[:10].squeeze())
```

    [[ 0.    0.    1.   ...,  0.    0.05  0.  ]
     [ 0.    0.    0.   ...,  0.    0.    0.  ]
     [ 0.    0.    1.   ...,  1.    0.    0.  ]
     ..., 
     [ 0.    0.    1.   ...,  1.    0.    0.  ]
     [ 0.    0.    1.   ...,  1.    0.    0.  ]
     [ 0.    0.    1.   ...,  1.    0.    0.  ]]



![png](output_3_1.png)



```python
x_train=trainx.squeeze()
x_test=testx.squeeze()
import sklearn
from sklearn.preprocessing import StandardScaler,MinMaxScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
mm=MinMaxScaler()
x_train=mm.fit_transform(x_train)
x_test=mm.transform(x_test)

x_train=np.expand_dims(x_train,2)
x_test=np.expand_dims(x_test,2)
```


```python
# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')

x=inputs
#x = Reshape((784, 1))(x)
x = Flatten()(x)
#x = Dense(featureDIM//2, activation='relu')(x)
x = Dense(28, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])



# instantiate encoder model
encoder = Model([inputs], [z_mean, z_log_var, z], name='encoder')
encoder.summary()

```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    encoder_input (InputLayer)      (None, 122, 1)       0                                            
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 122)          0           encoder_input[0][0]              
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 28)           3444        flatten_1[0][0]                  
    __________________________________________________________________________________________________
    z_mean (Dense)                  (None, 14)           406         dense_3[0][0]                    
    __________________________________________________________________________________________________
    z_log_var (Dense)               (None, 14)           406         dense_3[0][0]                    
    __________________________________________________________________________________________________
    z (Lambda)                      (None, 14)           0           z_mean[0][0]                     
                                                                     z_log_var[0][0]                  
    ==================================================================================================
    Total params: 4,256
    Trainable params: 4,256
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
plot_model(encoder, to_file='cvae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x =latent_inputs #keras.layers.concatenate([latent_inputs, y_labels])
x = Dense(28, activation='relu')(x)
#x = Dense(featureDIM//2, activation='relu')(x)
#x = Reshape((shape[1], shape[2], shape[3]))(x)

#for i in range(2):
#    x = Conv2DTranspose(filters=filters,
#                        kernel_size=kernel_size,
#                        activation='relu',
#                        strides=2,
#                        padding='same')(x)
#    filters //= 2

#outputs = Conv2DTranspose(filters=1,
#                          kernel_size=kernel_size,
#                          activation='sigmoid',
#                          padding='same',
#                          name='decoder_output')(x)

# instantiate decoder model
outputs = Dense(featureDIM, activation='sigmoid')(x)
decoder = Model([latent_inputs], outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='cvae_cnn_decoder.png', show_shapes=True)

# instantiate vae model
outputs = decoder([encoder([inputs])[2]])
cvae = Model([inputs], outputs, name='cvae')
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    z_sampling (InputLayer)      (None, 14)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 28)                420       
    _________________________________________________________________
    dense_5 (Dense)              (None, 122)               3538      
    =================================================================
    Total params: 3,958
    Trainable params: 3,958
    Non-trainable params: 0
    _________________________________________________________________



```python
models = (encoder, decoder)
data = (x_test, y_test)
```


```python
beta = 1.0
print("CVAE")
model_name = "cvae_cnn_mnist"
```

    CVAE



```python
reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
```


```python
reconstruction_loss *= image_size * image_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5 * beta
cvae_loss = K.mean(reconstruction_loss + kl_loss)
cvae.add_loss(cvae_loss)
cvae.compile(optimizer='rmsprop')
cvae.summary()
plot_model(cvae, to_file='cvae_cnn.png', show_shapes=True)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    encoder_input (InputLayer)   (None, 122, 1)            0         
    _________________________________________________________________
    encoder (Model)              [(None, 14), (None, 14),  4256      
    _________________________________________________________________
    decoder (Model)              (None, 122)               3958      
    =================================================================
    Total params: 8,214
    Trainable params: 8,214
    Non-trainable params: 0
    _________________________________________________________________



```python
cvae.fit([x_train],
                 epochs=1,
                 batch_size=batch_size,
                 validation_data=([x_test], None))
```

    Train on 125973 samples, validate on 22544 samples
    Epoch 1/1
    125973/125973 [==============================] - 7s 57us/step - loss: 69.8506 - val_loss: 44.1002





    <keras.callbacks.History at 0x7f179e8e1400>




```python
trainxVAE,_,_=encoder.predict([x_train])
testxVAE,_,_=encoder.predict([x_test])
```


```python
print(trainxVAE.shape)

```

    (125973, 14)



```python
def getMLP(pretrain=False,LengthOfInputSequences=14,trainable=True,lr=0.001):
    input1=keras.layers.Input(shape=(LengthOfInputSequences,))    
    
    kernel_size = 5
    filters = 32
    pool_size = 4
    lstm_output_size=512   
   
    #x1=Flatten()(x1)
    #x1=LSTM(512)(x1)
    #x1=Dropout(0.2)(x1)
    addLayer=Dense(1024,activation='relu')(input1)
    addLayer=Dropout(0.5)(addLayer)
    #addLayer=Dense(1024,activation='tanh')(addLayer)
    #addLayer=Dropout(0.5)(addLayer)
    
    output=Dense(5)(addLayer)
    output=Activation('softmax')(output)
    #output=Activation('sigmoid')(output)
    
    
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    seq.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=lr),metrics=['accuracy'])
    #seq.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=lr),metrics=['accuracy'])
    seq.summary()
    return seq

def getCNN(lr=0.001):
    cnn = Sequential()
    cnn.add(Convolution1D(64, 3, border_mode="same",activation="relu",input_shape=(8, 1)))
    cnn.add(Convolution1D(64, 3, border_mode="same", activation="relu"))
    cnn.add(MaxPooling1D(pool_length=(2)))
    cnn.add(Convolution1D(128, 3, border_mode="same", activation="relu"))
    cnn.add(Convolution1D(128, 3, border_mode="same", activation="relu"))
    cnn.add(MaxPooling1D(pool_length=(2)))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation="relu"))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(5, activation="sigmoid"))
    cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=lr),metrics=['accuracy'])
    return cnn

from keras import backend as K
K.clear_session()
md=getMLP(lr=0.0002)    #getCNNIMDB2()

plot_model(md, to_file='getMLP.png',show_shapes=True)
SVG(model_to_dot(md, show_shapes=True).create(prog='dot', format='svg'))   
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 14)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1024)              15360     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 5)                 5125      
    _________________________________________________________________
    activation_1 (Activation)    (None, 5)                 0         
    =================================================================
    Total params: 20,485
    Trainable params: 20,485
    Non-trainable params: 0
    _________________________________________________________________





![svg](output_14_1.svg)




```python
trainycat=keras.utils.to_categorical(trainy)
testycat=keras.utils.to_categorical(testy)
md.fit(trainxVAE,trainycat,validation_data=(testxVAE,testycat),batch_size=256,epochs=1)
```

    Train on 125973 samples, validate on 22544 samples
    Epoch 1/1
    125973/125973 [==============================] - 3s 24us/step - loss: 0.1274 - acc: 0.9632 - val_loss: 0.9500 - val_acc: 0.7299





    <keras.callbacks.History at 0x7f179ebe7588>




```python
predicts=md.predict(testxVAE)
predicts=np.argmax(predicts,axis=1)
print(predicts[:10])
print(classification_report([0 if s ==0 else 1 for s in testy],[0 if s ==0 else 1 for s in predicts],digits=4))
print(classification_report(testy,predicts,digits=4))
```

    [1 1 0 4 0 0 0 0 0 0]
                 precision    recall  f1-score   support
    
              0     0.6474    0.9720    0.7772      9711
              1     0.9659    0.5995    0.7398     12833
    
    avg / total     0.8287    0.7599    0.7559     22544
    
                 precision    recall  f1-score   support
    
              0     0.6474    0.9720    0.7772      9711
              1     0.9112    0.7521    0.8241      7460
              2     0.0000    0.0000    0.0000      2885
              3     0.0000    0.0000    0.0000        67
              4     0.7770    0.5799    0.6641      2421
    
    avg / total     0.6638    0.7299    0.6788     22544
    



```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
print(trainy[0],testy[0])
clf=LinearDiscriminantAnalysis()
#clf=XGBClassifier()
#clf=LinearSVC()
clf.fit(trainxVAE,trainy)
predicts=clf.predict(testxVAE)
print(classification_report([0 if s ==0 else 1 for s in testy],[0 if s ==0 else 1 for s in predicts],digits=4))
print(sklearn.metrics.accuracy_score([0 if s ==0 else 1 for s in testy],[0 if s ==0 else 1 for s in predicts]))
print(classification_report(testy,predicts,digits=4))

'''
对称VAE
             precision    recall  f1-score   support

          0       0.68      0.97      0.80      9711
          1       0.94      0.75      0.83      7460
          2       0.51      0.09      0.15      2885
          3       0.33      0.01      0.03        67
          4       0.76      0.68      0.71      2421

avg / total       0.75      0.75      0.72     22544

'''
```

    0 1
                 precision    recall  f1-score   support
    
              0     0.7080    0.9433    0.8089      9711
              1     0.9426    0.7057    0.8071     12833
    
    avg / total     0.8416    0.8080    0.8079     22544
    
    0.80801987225
                 precision    recall  f1-score   support
    
              0     0.7080    0.9433    0.8089      9711
              1     0.9446    0.7476    0.8346      7460
              2     0.9217    0.2856    0.4361      2885
              3     0.3000    0.0448    0.0779        67
              4     0.6831    0.7898    0.7326      2421
    
    avg / total     0.8098    0.7752    0.7593     22544
    





    '\n对称VAE\n             precision    recall  f1-score   support\n\n          0       0.68      0.97      0.80      9711\n          1       0.94      0.75      0.83      7460\n          2       0.51      0.09      0.15      2885\n          3       0.33      0.01      0.03        67\n          4       0.76      0.68      0.71      2421\n\navg / total       0.75      0.75      0.72     22544\n\n'




```python
for i in range(5):
    x1=x_test[i,:,:].squeeze()
    print(x1.shape)
    print(y_test[i])
    plt.figure()
    plt.imshow(x1)
```

    (28, 28)
    7
    (28, 28)
    2
    (28, 28)
    1
    (28, 28)
    0
    (28, 28)
    4



![png](output_18_1.png)



![png](output_18_2.png)



![png](output_18_3.png)



![png](output_18_4.png)



![png](output_18_5.png)



```python
alltesty=to_categorical(y_test)
zmeans,_,_=encoder.predict([x_test[0:3,:,:].reshape(3,featureDIM,1)])
print(zmeans.shape)
print(zmeans)
```

    (3, 2)
    [[ -1.06491208e+00  -3.02985024e+00]
     [  5.37734330e-01  -9.95900482e-04]
     [ -3.43391275e+00   1.21182609e+00]]



```python
x_decoded = decoder.predict([zmeans])
```


```python
print(x_decoded.shape)

```

    (3, 784)



```python
for x in x_decoded:
    plt.figure()
    plt.imshow(x.reshape(28,28).squeeze())
```


![png](output_22_0.png)



![png](output_22_1.png)



![png](output_22_2.png)



```python
# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# compute the number of labels
#um_labels = len(np.unique(y_train))

# network parameters
featureDIM=784
input_shape = (featureDIM,1)
#abel_shape = (num_labels, )
batch_size = 128
#ernel_size = 3
#ilters = 16
latent_dim = 2
epochs = 30
```


```python
import pickle
trainx=pickle.load(open('./trainx.pk','rb'))
trainy=pickle.load(open('./trainy.pk','rb'))
testx=pickle.load(open('./testx.pk','rb'))
testy=pickle.load(open('./testy.pk','rb'))
trainx=np.array(trainx).reshape(-1,122,1)
testx=np.array(testx).reshape(-1,122,1)
print(trainx.shape) #,trainy[0].shape)
print(testx.shape)
```

    (125973, 122, 1)
    (22544, 122, 1)

