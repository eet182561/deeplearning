#TestFile.py


# Below two lines are for fucking graphviz
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from models.VAE import VAE
from utils.loader import load_celeb_generator
vae = VAE((128,128,3),[32,64,64, 64]
                , [3,3,3,3]
                , [2,2,2,2]
                , [64,64,32,3]
                , [3,3,3,3]
                , [2,2,2,2]
                , 200)
a = load_celeb_generator('celeb',128,64)
vae.compile(0.005,1000)
vae.train_with_generator(a,2,10,'run')