import pickle

X = pickle.load(open('X.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

X = X/255

X = X.reshape(-1, 60, 60, 1)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from keras.callbacks import TensorBoard
import time

dense_layers = [3]
conv_layers = [3]
neurons = [64]


for dense_layer in dense_layers:
	for conv_layer in conv_layers:
		for neuron in neurons:

			NAME = '{}-denselayer-{}-convlayer-{}-neuron-{}'.format(dense_layer, conv_layer, neuron, int(time.time()))
			tensorboard = TensorBoard(log_dir = 'logs2\\{}'.format(NAME))


			model = Sequential()

			for l in range(conv_layer):
				model.add(Conv2D(neuron, (3,3), activation = 'relu'))
				model.add(MaxPooling2D((2,2)))

			model.add(Flatten())

			model.add(Dense(neuron, input_shape = X.shape[1:], activation = 'relu'))

			for l in range(dense_layer - 1):
				model.add(Dense(neuron, activation = 'relu'))

			model.add(Dense(2, activation = 'softmax'))

			model.compile(optimizer='adam',
			              loss='sparse_categorical_crossentropy',
			              metrics=['accuracy'])

			print('===================================================================================================================================')
			print('===================================================================================================================================')
			print('=========================================== RUNNING MODEL =========================================================================')
			print('=================================================='+ NAME + '======================================================================')
			print('===================================================================================================================================')
			print('===================================================================================================================================')

			model.fit(X, y, epochs=8, batch_size = 32, validation_split=0.1, callbacks = [tensorboard])

			model.save('3x3x64-catvsdog.model')