#TestFile.py


# Below two lines are for fucking graphviz
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from models.VAE import VAE
from utils.loader import load_celeb_generator,load_mnist
vae = VAE((28,28,1),[32,64,64]
                , [3,3,3]
                , [2,2,1]
                , [64,64,1]
                , [3,3,3]
                , [2,2,1]
                , 200)
(a,b),(c,d) = load_mnist()
vae.compile(0.005,1000)
vae.train(a,100,1024)