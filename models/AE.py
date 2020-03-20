# -*- coding: utf-8 -*-
"""
Convolutional Autoencoder model file with batchnorm and dropout support

Initialize the model with the following argument list
(   [number of convolutional filter in each layer],
    [filter sizes of filters at each layer],
    [strides in each layer],
    [number of decoder filters in each layer], 
    [decoder filter size at each layer],
    [strides at each layer],
    True for batchnorm layers,
    True for Dropout layers
)
from keras import losses to supply compile function with a loss function

"""

from keras import backend as K
from keras.layers import Conv2D,Dense,LeakyReLU,Flatten,BatchNormalization,Dropout,Input,Reshape,Conv2DTranspose,Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder():
    def __init__(self, input_dim, encoder_conv_filters, encoder_conv_size, encoder_conv_strides, decoder_conv_filters, decoder_conv_size, decoder_conv_strides, zdim, use_batch_norm=False, use_dropout=False):
        self.name = 'Autoencoder'
        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_filters
        self.decoder_conv_t_kernel_size = decoder_conv_size
        self.decoder_conv_t_strides = decoder_conv_strides
        self.z_dim = zdim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(self.encoder_conv_filters)
        self.n_layers_decoder = len(self.decoder_conv_t_filters)
        self.pred_calculated = False
        self._build()

    def _build(self):
        #Build the encoder
        encoder_input = Input(shape = self.input_dim,name = 'encoder_input')
        x=encoder_input
        for i in range(len(self.encoder_conv_filters)):
            conv_layer = Conv2D(self.encoder_conv_filters[i],kernel_size = self.encoder_conv_kernel_size[i],strides = self.encoder_conv_strides[i],padding='same',name = 'Encoder_conv_layer_'+str(i+1))
            x = conv_layer(x)
            x = LeakyReLU()(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            if self.use_dropout:
                x = Dropout(rate=0.4)(x)
            
        #After building encoder convs layer build the flatten and latent layer
        shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)
        encoder_output = Dense(self.z_dim,name = 'encoder_output')(x)
        self.encoder = Model(encoder_input,encoder_output,name='encoder')

        #Build the decoder
        decoder_input = Input(shape=(self.z_dim,),name='decoder_input')
        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = Reshape(shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            conv_t = Conv2DTranspose(filters = self.decoder_conv_t_filters[i],
                                    kernel_size=self.decoder_conv_t_kernel_size[i],
                                    strides=self.decoder_conv_t_strides[i],
                                    padding = 'same',
                                    name = 'decoder_conv_t_'+str(i+1)
                                    )
            x = conv_t(x)

            if i < self.n_layers_decoder-1:
                x = LeakyReLU()(x)
            if self.use_batch_norm:
                x = BatchNormalization()(x)
            if self.use_dropout:
                x = Dropout(rate=0.4)(x)
            else:
                x = Activation('sigmoid')(x)

        decoder_output = x
        self.decoder = Model(decoder_input,decoder_output,name = 'decoder')

        ### THE FULL AUTOENCODER
        model_input = encoder_input
        model_output = self.decoder(encoder_output)
        self.model = Model(model_input,model_output,name='Autoencoder')

    
    def compile(self,learning_rate,loss):
        ###COMPILING AUTOENCODER USING ADAM
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer,loss = loss)

    
    def displaymodel(self):
        self.encoder.summary()
        self.decoder.summary()

    
    def plotzdim2d(self,x,y,num_samples,pred_again=False):
        if (not self.pred_calculated) or pred_again:
            self.predict(x)
        
        ax = plt.figure(figsize=(12,15))
        plt.scatter(self.preds_enc[:num_samples,0],self.preds_enc[:num_samples,1],c=y[:num_samples])
        plt.show()

    def predict(self,x):
        self.preds = self.model.predict(x)
        self.preds_enc = self.encoder.predict(x)
        self.pred_calculated = True

    
    def plot_some(self,x,num_samples,pred_again=False):
        indices = np.random.choice(x.shape[0],num_samples)
        if (not self.pred_calculated) or pred_again:
            self.predict(x)
            
        for i,idx in enumerate(indices):
            img1 = x[idx]
            ax = plt.subplot(2,num_samples,i+1)
            ax.axis('off')
            ax.imshow(np.squeeze(img1))

            img2 = self.preds[idx]
            ax = plt.subplot(2,num_samples,i+1+num_samples)
            ax.axis('off')
            ax.imshow(np.squeeze(img2))
            
            
    def train(self,x_train,batch_size,shuffle,epochs,lr_decay,step_size=1):
        
        def lr_schedule(epoch,lr):
            new_lr = lr*(lr_decay**np.floor(epoch/step_size))
            #print('new_lr = ',new_lr)
            return new_lr
        
        lr_scheduler_callback = LearningRateScheduler(lr_schedule)
        callback_list = [lr_scheduler_callback]
        ##print(int(x_train.shape[0]//batch_size))
        self.model.fit(x_train,x_train,batch_size=32,callbacks=callback_list,epochs = epochs,shuffle=True)


