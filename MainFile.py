
from tensorflow.python.keras.callbacks import TensorBoard
from Model import Recognizer
from PIL import Image
import numpy as np
import time
import tensorflow as tf

data_dimension = 48

X1 = np.load( 'processed_data/x1.npy')
X2 = np.load( 'processed_data/x2.npy')
Y = np.load( 'processed_data/y.npy')

X1 = X1.reshape( ( X1.shape[0]  , data_dimension**2 * 3  ) ).astype( np.float32 )
X2 = X2.reshape( ( X2.shape[0]  , data_dimension**2 * 3  ) ).astype( np.float32 )

print( X1.shape )
print( X2.shape )
print( Y.shape )

recognizer = Recognizer()
#recognizer.load_model('models/model.h5')

parameters = {
    'batch_size' : 750 ,
    'epochs' : 5 ,
    'callbacks' : None , # [ TensorBoard( log_dir='logs/{}'.format( time.time() ) ) ] ,
    'val_data' : None
}

recognizer.fit( [ X1 , X2 ], Y, hyperparameters=parameters)
recognizer.save_model('models/model.h5')

print( Y[ 100 : 110 ] )
print( recognizer.predict( [ X1[100 : 110] , X2[ 100 : 110 ] ] ))


