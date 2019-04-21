
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

generator = datagen.flow_from_directory('images',target_size=(32,32),
										class_mode='categorical',
										batch_size=1 )
X = list()
Y = list()
count = 0
for inputs , outputs in generator:
	X.append( inputs[0] )
	Y.append( outputs[0] )
	count += 1
	if count > 70 :
		break

x = np.array( X )
y = np.array( Y )

samples_1 = list()
samples_2 = list()
labels = list()
for i in range(x.shape[0]):
	sample_1 = x[i]
	label_1 = y[i]
	for j in range(x.shape[0]):
		sample_2 = x[j]
		label_2 = y[j]
		samples_1.append( sample_1)
		samples_2.append( sample_2)
		if (label_1==label_2).all():
			labels.append( [1] )
		else:
			labels.append( [0] )

X1 = np.array( samples_1  )
X2 = np.array( samples_2 )
Y = np.array( labels )

print( X1.shape )
print( X2.shape )
print( Y.shape )

out_path = 'processed_data/'
np.save( '{}/x1.npy'.format( out_path ), X1 )
np.save( '{}/x2.npy'.format( out_path ), X2 )
np.save( '{}/y.npy'.format( out_path ) , Y )








