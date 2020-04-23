from keras.layers import Input,Conv2D,Dense,Activation,Dropout,Flatten,UpSampling2D,Conv2DTranspose,Reshape,BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop,Adam
from keras.losses import BinaryCrossentropy
import numpy as np
import matplotlib.pyplot as plt
import os

class GAN:
    def __init__(
                self,

                input_dim,
                
                discriminator_conv_filter,
                discriminator_conv_kernel_size,
                discriminator_conv_stride,
                disctriminator_activation_function,
                discriminator_learning_rate,
                discriminator_batch_norm_momentum,
                discriminator_dropout_rate,

                generator_initial_dense_layer_size,
                generator_upsample,
                generator_conv_filter,
                generator_conv_kernel_size,
                generator_conv_strides,
                generator_batch_norm_momentum,
                genrator_activation,
                genrator_dropout_rate,
                generator_learning_rate,
                optimiser,
                zdim
                ):


                self.input_dim = input_dim
            
                self.discriminator_conv_filter = discriminator_conv_filter
                self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
                self.discriminator_conv_stride = discriminator_conv_stride
                self.disctriminator_activation_function = disctriminator_activation_function
                self.discriminator_learning_rate = discriminator_learning_rate
                self.discriminator_batch_norm_momentum =discriminator_batch_norm_momentum
                self.discriminator_dropout_rate =discriminator_dropout_rate

                self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
                self.generator_upsample = generator_upsample
                self.generator_conv_filter = generator_conv_filter
                self.generator_conv_kernel_size = generator_conv_kernel_size
                self.generator_conv_strides = generator_conv_strides
                self.generator_batch_norm_momentum = generator_batch_norm_momentum
                self.genrator_activation = genrator_activation
                self.genrator_dropout_rate = genrator_dropout_rate
                self.generator_learning_rate = generator_learning_rate
                self.optimiser = optimiser
                self.zdim = zdim

                self.d_losses = []
                self.g_losses = []
                
                self.epoch = 0 

                self._build_discriminator()
                self._build_generator()
                self._build_adverserial()

    def _build_discriminator(self):
        # Discriminator
        discriminator_input = Input(shape=self.input_dim,name = 'discriminator_input')
        print('input dim',self.input_dim)
        x = discriminator_input
        
        for i in range(len(self.discriminator_conv_filter)):
            conv = Conv2D(filters = self.discriminator_conv_filter[i],
                        kernel_size=self.discriminator_conv_kernel_size[i],
                        padding='same',
                        strides=self.discriminator_conv_stride[i],
                        name = 'Discriminator_conv_layer_'+str(i+1))
            x = conv(x)
            if self.discriminator_batch_norm_momentum : 
                bn = BatchNormalization(momentum = self.discriminator_batch_norm_momentum,name = 'Batch_norm_layer_'+str(i+1))
                x = bn(x)
            x = Activation(self.disctriminator_activation_function)(x)
            if self.discriminator_dropout_rate:
                x = Dropout(self.discriminator_dropout_rate)(x)
        x = Flatten()(x)        
        discriminator_output = Dense(1,activation='sigmoid')(x)
        self.discriminator = Model(discriminator_input,discriminator_output)

    def get_opti(self, lr):
        if self.optimiser == 'adam':
            opti = Adam(lr=lr, beta_1=0.5)
        elif self.optimiser == 'rmsprop':
            opti = RMSprop(lr=lr)
        else:
            opti = Adam(lr=lr)
        return opti

    def _build_generator(self):
        generator_input = Input(shape=(self.zdim,),name = 'generator_input')
        x = generator_input
        x = Dense(np.prod(self.generator_initial_dense_layer_size))(x)
        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
        x = Activation(self.genrator_activation)(x)
        x = Reshape(self.generator_initial_dense_layer_size)(x)
        if self.genrator_dropout_rate:
            x = Dropout(rate = self.genrator_dropout_rate)(x)
        for i in range(len(self.generator_conv_filter)):
            if self.generator_upsample[i]==2:
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters=self.generator_conv_filter[i],
                    kernel_size = self.generator_conv_kernel_size[i],
                    padding = 'same',
                    name = 'generator_cov_'+str(i)
                    )(x)
            else:
                x = Conv2DTranspose(
                    filters = self.generator_conv_filter[i]
                    , kernel_size = self.generator_conv_kernel_size[i]
                    , padding = 'same'
                    , strides = self.generator_conv_strides[i]
                    , name = 'generator_conv_' + str(i)
                    )(x)

            if i<len(self.generator_conv_filter)-1:
                if self.generator_batch_norm_momentum:
                    x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)
                x = Activation(self.genrator_activation)(x)
            else:
                x = Activation('tanh')(x)

            generator_output = x
            self.generator = Model(generator_input,generator_output,name='generator')
        
    def set_trainable(self,model,value):
        model.trainable = value
        for layer in model.layers:
            layer.trainable = value

    def _build_adverserial(self):
        # Training the discriminator

        self.discriminator.compile(
            optimizer = RMSprop(lr = 0.0008),
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )

        self.set_trainable(self.discriminator,False)

        model_input = Input(shape=(self.zdim,),name = 'model_input')
        model_output = self.discriminator(self.generator(model_input))
        self.model = Model(model_input,model_output)
        self.model.compile(
            optimizer = self.get_opti(self.generator_learning_rate),
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )
        self.set_trainable(self.discriminator,True)


    def train_discriminator(self,x_train,batch_size,using_generator):
        #Prepare labels
        valid = np.ones(shape=(batch_size,1))
        fake = np.zeros(shape=(batch_size,1))
        
        #prepare real iamges
        if using_generator:
            true_images = next(x_train) #next return (x,y); 0 for x
            if true_images.shape[0] != batch_size:
                true_images = next(x_train)[0]
        else:
            idx = np.random.randint(0,x_train.shape[0],size = batch_size)
            true_images = x_train[idx]

        #prepare fake ones generated by generator
        noise = np.random.normal(0,1,size = (batch_size,self.zdim))
        fake_images = self.generator.predict(noise)
        d_loss_true,d_acc_true = self.discriminator.train_on_batch(true_images,valid)
        d_loss_fake,d_acc_fake = self.discriminator.train_on_batch(fake_images,fake)

        d_loss = 0.5 *(d_loss_true + d_loss_fake)
        d_acc = 0.5*(d_acc_fake + d_acc_true)
        
        return [d_loss,d_loss_true,d_loss_fake,d_acc,d_acc_true,d_acc_fake]

    def train_generator(self,batch_size):
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0,1,(batch_size,self.zdim))
        return self.model.train_on_batch(noise,valid)

    def sample_images(self,run_folder):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.zdim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15,15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()
        
    def train(self,x_train,batch_size,epochs,run_folder,using_generator,print_every_n_batch):
        for epoch in range(self.epoch,epochs):
            d = self.train_discriminator(x_train,batch_size,using_generator)
            g = self.train_generator(batch_size)

            print("%d ,[D Loss=%.3f,%.3f,%.3f] [D acc = %.3f,%.3f,%.3f] [G loss = %.3f} G acc = %.3f}]"%(self.epoch,d[0],d[1],d[2],d[3],d[4],d[5],g[0],g[1]))
            self.d_losses.append(d)
            self.g_losses.append(g)
            g_best = None 
            if (epoch)%print_every_n_batch == 0:
                self.sample_images(run_folder)
                self.model.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (epoch)))
                if g_best is None or g[0]<g_best:
                    self.model.save_weights(os.path.join(run_folder, 'weights/weights_best_at_epoch_{self.epoch:%5d}.h5'))
                    if g_best is not None:
                        os.remove(os.path.join(run_folder, 'weights/weights_best_at_epoch_{g_best:%5d}.h5'))
                    g_best = self.epoch 
            self.epoch=epoch+1