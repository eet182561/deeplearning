# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 11:43:16 2020

@author: dell

Variational autoencoder model
"""

import numpy as np

import matplotlib.pyplot as plt

from keras.layers import Conv2D,Dense,Input,BatchNormalization,Dropout
from keras.layers import LeakyReLU,Activation,Flatten,Lambda,Reshape,Conv2DTranspose

from keras.models import Model

from keras import optimizers

from keras import backend as K

from keras.utils import plot_model

from utils.callbacks import return_default_callbacks_list,Custom_callback,step_decay_schedule,checkpoints

import pickle

import os

class VAE:
    def __init__(self,image_size,encoder_filters,encoder_kernel_size,
                 encoder_kernel_strides,decoder_filters,decoder_kernel_size,
                 decoder_kernel_strides,zdim,use_batchnorm=False,
                 bn_momentum=None,use_dropout=False,dropout_rate=None):
        
        self.image_size = image_size
        self.name = 'Variational autoencoder'
        self.encoder_filters = encoder_filters
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_kernel_strides = encoder_kernel_strides
        self.decoder_filters = decoder_filters
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_kernel_strides = decoder_kernel_strides
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.encoder_len = len(self.encoder_filters)
        self.decoder_len = len(self.decoder_filters)
        self.bn_momentum = bn_momentum
        self.dropout_rate = dropout_rate
        self.zdim = zdim
        self._build()
    
    
    def _build(self):
        #Encoder
        self.encoder_input = Input(shape = self.image_size,
                                   name='Encoder_input')
        
        x = self.encoder_input
        
        for i in range(self.encoder_len):
            conv_layer = Conv2D(filters = self.encoder_filters[i],
                                kernel_size = self.encoder_kernel_size[i],
                                strides = self.encoder_kernel_strides[i],
                                name = 'ConvolutionLayer_' + str(i+1),
                                padding = 'same')
            x = conv_layer(x)
            
            if self.use_batchnorm:
                bn = BatchNormalization(momentum = self.bn_momentum,
                                        name = 'BN_' + str(i+1))
                x = bn(x)
                
            x = LeakyReLU()(x)
            
            if self.use_dropout:
                dp = Dropout(rate = self.dropout_rate,
                             name = 'Dropout_' + str(i+1))
                x = dp(x)
        
        self.shape_before_flattening = K.int_shape(x)[1:]
        x = Flatten()(x)
        self.mu = Dense(units = self.zdim,name = 'mu')(x)
        self.log_var = Dense(units = self.zdim,name = 'log_var')(x)
        encoder_mu_log_var = Model(self.encoder_input,
                                   [self.mu,self.log_var],
                                   name='encoder mu log var')
        
        def sampling(args):
            mu,log_var = args
            epsilon = K.random_normal(shape=K.shape(mu),mean = 0.,stddev = 1.)
            return mu + K.exp(log_var/2)*epsilon
        
        self.encoder_output = Lambda(sampling,name='encoder_output')([self.mu,self.log_var])
        
        self.encoder = Model(self.encoder_input,self.encoder_output,name='Encoder')
        
        
        #Decoder
        
        self.decoder_input = Input(shape=(self.zdim,),name = 'decoder_input')
        
        x = Dense(np.prod(self.shape_before_flattening))(self.decoder_input)
        x = Reshape(self.shape_before_flattening)(x)
        
        for i in range(self.decoder_len):
            conv_t = Conv2DTranspose(filters = self.decoder_filters[i],
                                     kernel_size = self.decoder_kernel_size[i],
                                     strides = self.decoder_kernel_strides[i],
                                     padding = 'same', name = 'conv2dt_' + str(i+1))
            x = conv_t(x)
            
            if i < self.decoder_len - 1 :   
                if self.use_batchnorm:
                    bn = BatchNormalization(momentum = self.bn_momentum,name = 'bn_'+str(i+1))
                    x = bn(x)
                    
                x = LeakyReLU()(x)
                
                if self.use_dropout:
                    dp = Dropout(self.dropout_rate,name = 'dp_'+str(i+1))
                    x = dp(x)
            else :
                x = Activation('sigmoid')(x)
                
        self.decoder_output = x
        self.decoder = Model(self.decoder_input,self.decoder_output)
        self.VAE_output = self.decoder(self.encoder_output)
        self.model = Model(self.encoder_input,self.VAE_output)
        
        
    def compile(self,learning_rate,r_loss_factor):
        optimizer = optimizers.Adam(lr = learning_rate)
        self.lr = learning_rate
        def vae_r_loss(y_true,y_pred):
            r_loss = K.mean(K.square(y_true-y_pred),axis = [1,2,3])
            return r_loss_factor*r_loss
        
        def vae_kl_loss(y_true,y_pred):
            kl_loss = -0.5*K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var),axis = 1)
            return kl_loss
        
        def vae_loss(y_true,y_pred):
            r_loss = vae_r_loss(y_true,y_pred)
            kl_loss = vae_kl_loss(y_true,y_pred)
            return r_loss + kl_loss
        
        self.model.compile(optimizer = optimizer, loss = vae_loss,metrics = [vae_r_loss,vae_kl_loss])
        
        
        
    def train(self,x_train,epochs,bs):
        #use with mnist
        callback_list = return_default_callbacks_list('run',1000,0,self,0.0005,1,1)
        self.model.fit(x=x_train,y=x_train,batch_size = bs,callbacks = callback_list,epochs=epochs,shuffle = True)
    
    def train_with_generator(self,data_flow, epochs, steps_per_epoch, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1):
        checkpoints_list = checkpoints(run_folder)
        lrschd = step_decay_schedule(self.lr,lr_decay,steps_per_epoch)
        cblist = checkpoints_list + lrschd + [Custom_callback(run_folder,print_every_n_batches,initial_epoch,self)]
        self.save(run_folder)
        if not os.path.exists(os.path.join(run_folder,'weights')):
            os.mkdir(os.path.join(run_folder,'weights'))
        self.model.save_weights(os.path.join(run_folder, 'weights\weights.h5'))
        
        self.model.fit_generator(
            data_flow
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
            , callbacks = cblist
            , steps_per_epoch=steps_per_epoch 
            )
        
    def save(self, folder):

        if not os.path.exists(os.path.join(folder,'images')):
            print('making folders')
            os.makedirs(folder,exist_ok=True)
            os.makedirs(os.path.join(folder, 'viz'),exist_ok=True)
            os.makedirs(os.path.join(folder, 'weights'),exist_ok=True)
            os.makedirs(os.path.join(folder, 'images'),exist_ok=True)
            print('folder made')
            
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.image_size
                , self.encoder_filters
                , self.encoder_kernel_size
                , self.encoder_kernel_strides
                , self.decoder_filters
                , self.decoder_kernel_size
                , self.decoder_kernel_strides
                , self.zdim
                , self.use_batchnorm
                , self.use_dropout
                ], f)

        self.plot_model(folder)
    
    def plot_model(self,run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.encoder, to_file=os.path.join(run_folder ,'viz/encoder.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.decoder, to_file=os.path.join(run_folder ,'viz/decoder.png'), show_shapes = True, show_layer_names = True)
        
    def load_weights(self,filepath):
        self.model.load_weights(filepath)
