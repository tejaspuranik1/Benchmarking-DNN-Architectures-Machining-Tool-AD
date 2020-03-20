# MLP
# source: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py
import keras

class Classifier_MLP:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
		self.output_directory = output_directory
		self.model = self.build_model(input_shape, nb_classes)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		# flatten/reshape because when multivariate all should be on the same axis
		input_layer_flattened = keras.layers.Flatten()(input_layer)

		layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
		layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

		layer_2 = keras.layers.Dropout(0.2)(layer_1)
		layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

		layer_3 = keras.layers.Dropout(0.2)(layer_2)
		layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

		output_layer = keras.layers.Dropout(0.3)(layer_3)
		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
			metrics=['accuracy'])

		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

		file_path = self.output_directory+'best_model.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
			save_best_only=True)

		self.callbacks = [reduce_lr,model_checkpoint]

		return model

	def fit(self, x_train, y_train, x_val, y_val,y_true):
		# x_val and y_val are only used to monitor the test loss and NOT for training
		batch_size = 16
		nb_epochs = 100

		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))


		self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)


		keras.backend.clear_session()