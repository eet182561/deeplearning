# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 23:46:15 2020
Data Loader file :currently for celebA dataset
@author: dell
"""




'''
celeb a dataset is huge so we will load it using a generator
'''
from keras.preprocessing.image import ImageDataGenerator


'''
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False, 
samplewise_center=False, featurewise_std_normalization=False, 
samplewise_std_normalization=False, zca_whitening=False, 
zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0, 
height_shift_range=0.0, brightness_range=None, shear_range=0.0, 
zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', 
cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, 
preprocessing_function=None, data_format='channels_last', 
validation_split=0.0, interpolation_order=1, dtype='float32')
'''
import os

def load_celeb_generator(folder_name,image_size,batch_size):
    data_folder = os.path.join('./data',folder_name)   ## For unix use / for windows use \
    if not os.path.exists(data_folder):
        print("Error")
        return
    print("Path exists. Continuing")
    data_gen = ImageDataGenerator(preprocessing_function=lambda x:(x.astype('float32')-127.5)/127.5 )
    
    x_train = data_gen.flow_from_directory(data_folder
                                           ,target_size = (image_size,image_size)
                                           ,batch_size = batch_size
                                           ,shuffle = True
                                           ,class_mode = 'input'
                                           ,subset = 'training')
    return x_train


from keras.datasets import mnist
from keras.utils import to_categorical

def load_mnist():
    #return numpy ndarray
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train,y_train),(x_test,y_test)

def load_mnist_generator():
    #returns iterators to be used with generators
    '''
    Trick to increase the dimension
    we have x_train[samples,height,width]
    x_train[samples,height,width,1] is what we want
    we can do this in 3 ways:
        using np.newaxis
        x_train[:,:,:,np.newaxis]
        using np.expand_dims()
        x_train = np.expand_dims(x_train,axis=3)
        using reshape()
        x_train = x_train.reshape(*x.shape,2) <---* is for unpacking the shape tuple
    '''
    datagen = ImageDataGenerator(rescale=1./255)
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    train_iter = datagen.flow(x_train,to_categorical(y_train),32,shuffle = True)
    test_iter = datagen.flow(x_test,to_categorical(y_test),batch_size=32,shuffle=True)
    return train_iter,test_iter
    