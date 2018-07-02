

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

    Using TensorFlow backend.



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
# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')

x=inputs
#x = Reshape((784, 1))(x)
x = Flatten()(x)
x = Dense(featureDIM//2, activation='relu')(x)
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
    encoder_input (InputLayer)      (None, 784, 1)       0                                            
    __________________________________________________________________________________________________
    flatten_3 (Flatten)             (None, 784)          0           encoder_input[0][0]              
    __________________________________________________________________________________________________
    dense_18 (Dense)                (None, 392)          307720      flatten_3[0][0]                  
    __________________________________________________________________________________________________
    z_mean (Dense)                  (None, 2)            786         dense_18[0][0]                   
    __________________________________________________________________________________________________
    z_log_var (Dense)               (None, 2)            786         dense_18[0][0]                   
    __________________________________________________________________________________________________
    z (Lambda)                      (None, 2)            0           z_mean[0][0]                     
                                                                     z_log_var[0][0]                  
    ==================================================================================================
    Total params: 309,292
    Trainable params: 309,292
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
plot_model(encoder, to_file='cvae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x =latent_inputs #keras.layers.concatenate([latent_inputs, y_labels])
x = Dense(featureDIM//2, activation='relu')(x)
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
    z_sampling (InputLayer)      (None, 2)                 0         
    _________________________________________________________________
    dense_19 (Dense)             (None, 392)               1176      
    _________________________________________________________________
    dense_20 (Dense)             (None, 784)               308112    
    =================================================================
    Total params: 309,288
    Trainable params: 309,288
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
    encoder_input (InputLayer)   (None, 784, 1)            0         
    _________________________________________________________________
    encoder (Model)              [(None, 2), (None, 2), (N 309292    
    _________________________________________________________________
    decoder (Model)              (None, 784)               309288    
    =================================================================
    Total params: 618,580
    Trainable params: 618,580
    Non-trainable params: 0
    _________________________________________________________________


    /usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:7: UserWarning: Output "decoder" missing from loss dictionary. We assume this was done on purpose, and we will not be expecting any data to be passed to "decoder" during training.
      import sys



```python
cvae.fit([x_train.reshape(60000,featureDIM,1)],
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=([x_test.reshape(10000,featureDIM,1)], None))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/30
    60000/60000 [==============================] - 4s 61us/step - loss: 191.7117 - val_loss: 172.1673
    Epoch 2/30
    60000/60000 [==============================] - 3s 56us/step - loss: 169.7340 - val_loss: 167.8803
    Epoch 3/30
    60000/60000 [==============================] - 3s 55us/step - loss: 166.2940 - val_loss: 165.2266
    Epoch 4/30
    60000/60000 [==============================] - 3s 56us/step - loss: 164.1443 - val_loss: 163.3621
    Epoch 5/30
    60000/60000 [==============================] - 3s 55us/step - loss: 162.6151 - val_loss: 162.5887
    Epoch 6/30
    60000/60000 [==============================] - 3s 56us/step - loss: 161.4213 - val_loss: 161.1790
    Epoch 7/30
    60000/60000 [==============================] - 3s 56us/step - loss: 160.3922 - val_loss: 160.0977
    Epoch 8/30
    60000/60000 [==============================] - 3s 56us/step - loss: 159.4709 - val_loss: 159.4454
    Epoch 9/30
    60000/60000 [==============================] - 3s 55us/step - loss: 158.5479 - val_loss: 158.7440
    Epoch 10/30
    60000/60000 [==============================] - 3s 55us/step - loss: 157.7116 - val_loss: 157.7463
    Epoch 11/30
    60000/60000 [==============================] - 3s 56us/step - loss: 156.9934 - val_loss: 157.0535
    Epoch 12/30
    60000/60000 [==============================] - 3s 56us/step - loss: 156.3357 - val_loss: 156.6330
    Epoch 13/30
    60000/60000 [==============================] - 3s 55us/step - loss: 155.7335 - val_loss: 156.1049
    Epoch 14/30
    60000/60000 [==============================] - 3s 56us/step - loss: 155.2283 - val_loss: 155.9004
    Epoch 15/30
    60000/60000 [==============================] - 3s 56us/step - loss: 154.7503 - val_loss: 155.6825
    Epoch 16/30
    60000/60000 [==============================] - 3s 56us/step - loss: 154.3269 - val_loss: 154.9596
    Epoch 17/30
    60000/60000 [==============================] - 3s 56us/step - loss: 153.9384 - val_loss: 155.1268
    Epoch 18/30
    60000/60000 [==============================] - 3s 56us/step - loss: 153.6065 - val_loss: 154.3794
    Epoch 19/30
    60000/60000 [==============================] - 3s 57us/step - loss: 153.2833 - val_loss: 154.2342
    Epoch 20/30
    60000/60000 [==============================] - 3s 55us/step - loss: 152.9803 - val_loss: 154.0863
    Epoch 21/30
    60000/60000 [==============================] - 3s 56us/step - loss: 152.6909 - val_loss: 153.8598
    Epoch 22/30
    60000/60000 [==============================] - 3s 54us/step - loss: 152.4342 - val_loss: 153.8540
    Epoch 23/30
    60000/60000 [==============================] - 3s 55us/step - loss: 152.2214 - val_loss: 153.5750
    Epoch 24/30
    60000/60000 [==============================] - 3s 56us/step - loss: 151.9749 - val_loss: 153.0319
    Epoch 25/30
    60000/60000 [==============================] - 3s 57us/step - loss: 151.7746 - val_loss: 153.6970
    Epoch 26/30
    60000/60000 [==============================] - 3s 56us/step - loss: 151.5246 - val_loss: 153.0640
    Epoch 27/30
    60000/60000 [==============================] - 3s 55us/step - loss: 151.3595 - val_loss: 153.6995
    Epoch 28/30
    60000/60000 [==============================] - 3s 56us/step - loss: 151.1718 - val_loss: 152.7538
    Epoch 29/30
    60000/60000 [==============================] - 3s 56us/step - loss: 150.9929 - val_loss: 152.4718
    Epoch 30/30
    60000/60000 [==============================] - 3s 56us/step - loss: 150.8450 - val_loss: 153.1941





    <keras.callbacks.History at 0x7fde605f0ac8>




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



![png](output_9_1.png)



![png](output_9_2.png)



![png](output_9_3.png)



![png](output_9_4.png)



![png](output_9_5.png)



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


![png](output_13_0.png)



![png](output_13_1.png)



![png](output_13_2.png)

