

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


```python
# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# compute the number of labels
num_labels = len(np.unique(y_train))

# network parameters
input_shape = (image_size, image_size, 1)
label_shape = (num_labels, )
batch_size = 128
kernel_size = 3
filters = 16
latent_dim = 2
epochs = 30
```


```python
# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
#y_labels = Input(shape=label_shape, name='class_labels')
#x = Dense(image_size * image_size)(y_labels)
#x = Reshape((image_size, image_size, 1))(x)
#x = keras.layers.concatenate([inputs, x])
x=inputs
for i in range(2):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model([inputs], [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='cvae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x =latent_inputs #keras.layers.concatenate([latent_inputs, y_labels])
x = Dense(shape[1]*shape[2]*shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model([latent_inputs], outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='cvae_cnn_decoder.png', show_shapes=True)

# instantiate vae model
outputs = decoder([encoder([inputs])[2]])
cvae = Model([inputs], outputs, name='cvae')
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    encoder_input (InputLayer)      (None, 28, 28, 1)    0                                            
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 14, 14, 32)   320         encoder_input[0][0]              
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 7, 7, 64)     18496       conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    flatten_7 (Flatten)             (None, 3136)         0           conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    dense_14 (Dense)                (None, 16)           50192       flatten_7[0][0]                  
    __________________________________________________________________________________________________
    z_mean (Dense)                  (None, 2)            34          dense_14[0][0]                   
    __________________________________________________________________________________________________
    z_log_var (Dense)               (None, 2)            34          dense_14[0][0]                   
    __________________________________________________________________________________________________
    z (Lambda)                      (None, 2)            0           z_mean[0][0]                     
                                                                     z_log_var[0][0]                  
    ==================================================================================================
    Total params: 69,076
    Trainable params: 69,076
    Non-trainable params: 0
    __________________________________________________________________________________________________
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    z_sampling (InputLayer)      (None, 2)                 0         
    _________________________________________________________________
    dense_15 (Dense)             (None, 3136)              9408      
    _________________________________________________________________
    reshape_8 (Reshape)          (None, 7, 7, 64)          0         
    _________________________________________________________________
    conv2d_transpose_13 (Conv2DT (None, 14, 14, 64)        36928     
    _________________________________________________________________
    conv2d_transpose_14 (Conv2DT (None, 28, 28, 32)        18464     
    _________________________________________________________________
    decoder_output (Conv2DTransp (None, 28, 28, 1)         289       
    =================================================================
    Total params: 65,089
    Trainable params: 65,089
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
    encoder_input (InputLayer)   (None, 28, 28, 1)         0         
    _________________________________________________________________
    encoder (Model)              [(None, 2), (None, 2), (N 69076     
    _________________________________________________________________
    decoder (Model)              (None, 28, 28, 1)         65089     
    =================================================================
    Total params: 134,165
    Trainable params: 134,165
    Non-trainable params: 0
    _________________________________________________________________


    /usr/local/lib/python3.5/dist-packages/ipykernel_launcher.py:7: UserWarning: Output "decoder" missing from loss dictionary. We assume this was done on purpose, and we will not be expecting any data to be passed to "decoder" during training.
      import sys



```python
cvae.fit([x_train],
                 epochs=epochs,
                 batch_size=batch_size,
                 validation_data=([x_test], None))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/30
    60000/60000 [==============================] - 7s 116us/step - loss: 199.7351 - val_loss: 175.3303
    Epoch 2/30
    60000/60000 [==============================] - 6s 107us/step - loss: 167.8911 - val_loss: 163.2075
    Epoch 3/30
    60000/60000 [==============================] - 6s 106us/step - loss: 162.8290 - val_loss: 162.2069
    Epoch 4/30
    60000/60000 [==============================] - 6s 106us/step - loss: 159.9992 - val_loss: 159.0820
    Epoch 5/30
    60000/60000 [==============================] - 6s 106us/step - loss: 158.1391 - val_loss: 157.6908
    Epoch 6/30
    60000/60000 [==============================] - 6s 106us/step - loss: 156.7073 - val_loss: 155.6943
    Epoch 7/30
    60000/60000 [==============================] - 6s 106us/step - loss: 155.5056 - val_loss: 154.9447
    Epoch 8/30
    60000/60000 [==============================] - 6s 105us/step - loss: 154.4924 - val_loss: 154.3607
    Epoch 9/30
    60000/60000 [==============================] - 6s 106us/step - loss: 153.6142 - val_loss: 153.0364
    Epoch 10/30
    60000/60000 [==============================] - 6s 106us/step - loss: 152.8341 - val_loss: 152.7628
    Epoch 11/30
    60000/60000 [==============================] - 6s 106us/step - loss: 152.1322 - val_loss: 151.4818
    Epoch 12/30
    60000/60000 [==============================] - 6s 106us/step - loss: 151.4743 - val_loss: 150.8764
    Epoch 13/30
    60000/60000 [==============================] - 6s 107us/step - loss: 150.8209 - val_loss: 150.9482
    Epoch 14/30
    60000/60000 [==============================] - 6s 105us/step - loss: 150.3562 - val_loss: 151.6314
    Epoch 15/30
    60000/60000 [==============================] - 6s 107us/step - loss: 149.8189 - val_loss: 150.0256
    Epoch 16/30
    60000/60000 [==============================] - 6s 106us/step - loss: 149.4247 - val_loss: 149.5010
    Epoch 17/30
    60000/60000 [==============================] - 6s 106us/step - loss: 149.1117 - val_loss: 150.0455
    Epoch 18/30
    60000/60000 [==============================] - 6s 103us/step - loss: 148.7119 - val_loss: 150.0130
    Epoch 19/30
    60000/60000 [==============================] - 6s 104us/step - loss: 148.4299 - val_loss: 150.3348
    Epoch 20/30
    60000/60000 [==============================] - 6s 107us/step - loss: 148.1072 - val_loss: 148.4456
    Epoch 21/30
    60000/60000 [==============================] - 6s 105us/step - loss: 147.8161 - val_loss: 148.1030
    Epoch 22/30
    60000/60000 [==============================] - 6s 104us/step - loss: 147.5517 - val_loss: 148.3302
    Epoch 23/30
    60000/60000 [==============================] - 6s 106us/step - loss: 147.3422 - val_loss: 152.3048
    Epoch 24/30
    60000/60000 [==============================] - 6s 105us/step - loss: 147.1328 - val_loss: 148.5181
    Epoch 25/30
    60000/60000 [==============================] - 6s 106us/step - loss: 146.9007 - val_loss: 147.5294
    Epoch 26/30
    60000/60000 [==============================] - 6s 106us/step - loss: 146.7300 - val_loss: 147.9071
    Epoch 27/30
    60000/60000 [==============================] - 6s 105us/step - loss: 146.5411 - val_loss: 147.2750
    Epoch 28/30
    60000/60000 [==============================] - 6s 106us/step - loss: 146.3404 - val_loss: 147.7102
    Epoch 29/30
    60000/60000 [==============================] - 6s 107us/step - loss: 146.1385 - val_loss: 146.8977
    Epoch 30/30
    60000/60000 [==============================] - 6s 106us/step - loss: 145.9823 - val_loss: 147.4093





    <keras.callbacks.History at 0x7f83884f7c18>




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



![png](output_8_1.png)



![png](output_8_2.png)



![png](output_8_3.png)



![png](output_8_4.png)



![png](output_8_5.png)



```python
alltesty=to_categorical(y_test)
zmeans,_,_=encoder.predict([x_test[0:3,:,:]])
print(zmeans.shape)
print(zmeans)
```

    (3, 2)
    [[-1.30963635  2.4822998 ]
     [ 0.41578639  0.13433978]
     [-2.04411769 -1.65763581]]



```python
x_decoded = decoder.predict([zmeans])
```


```python
print(x_decoded.shape)

```

    (3, 28, 28, 1)



```python
for x in x_decoded:
    plt.figure()
    plt.imshow(x.squeeze())
```


![png](output_12_0.png)



![png](output_12_1.png)



![png](output_12_2.png)

