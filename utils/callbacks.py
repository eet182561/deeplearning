# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:32:41 2020

@author: dell
"""

from keras.callbacks import Callback, LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
import os


class Custom_callback(Callback):
    def __init__(self,run_folder,print_every_n_batches,initial_epoch,vae):
        self.vae = vae
        self.print_every_n_batches = print_every_n_batches
        self.initial_epoch = initial_epoch
        self.run_folder = run_folder
        
    def on_batch_end(self,batch,logs={}):
        if batch % self.print_every_n_batches == 0:
            z_new = np.random.normal(size = (1,self.vae.zdim))
            reconst = self.vae.decoder.predict(np.array(z_new))[0].squeeze()
            
            filepath = os.path.join(self.run_folder, 'images', 'img_' + str(self.epoch).zfill(3) + '_' + str(batch) + '.jpg')
            if len(reconst.shape) == 2:
                plt.imsave(filepath, reconst, cmap='gray_r')
            else:
                plt.imsave(filepath, reconst)
    def on_epoch_begin(self,epoch,logs={}):
        self.epoch=epoch+1


def step_decay_schedule(initial_lr,decay_rate,step_size=1):
    '''
    Wrapper function to get the input values of decay rate and step size
    '''
    
    def schedule(epoch):
        new_lr  = initial_lr*(decay_rate**np.floor(epoch/step_size))
        return new_lr
    
    return [LearningRateScheduler(schedule)]



from keras.callbacks import ModelCheckpoint

def checkpoints(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    checkpoint_loc1 = os.path.join(folder,'/weights/weights-{epoch:03d}-{loss:.2f}.hd5')
    checkpoint_best = os.path.join(folder,'weights/weights.h5')
    checkpoint1 = ModelCheckpoint(checkpoint_loc1,save_weights_only=True,verbose = 1)
    checkpoint2 = ModelCheckpoint(checkpoint_best,monitor='loss',save_best_only=True,save_weights_only = True,verbose = 1)
    return [checkpoint1,checkpoint2]

def return_default_callbacks_list(folder,print_every_n_batches,initial_epoch,vae,initial_lr,decay_rate,step_size=1):
    return step_decay_schedule(initial_lr,decay_rate,step_size)+checkpoints(folder)+[custom_callback(folder,print_every_n_batches,initial_epoch,vae)]
