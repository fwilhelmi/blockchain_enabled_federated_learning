import tensorflow as tf
import extra_keras_datasets.emnist as centr_emnist  # See https://github.com/machinecurve/extra_keras_datasets

# Load the centralized version of the dataset
(x_train, y_train), (x_test, y_test) = centr_emnist.load_data(type='digits')
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

# Define the model (same as for the federated setting in EMNIST)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Input(shape=(28, 28, 1))),
model.add(tf.keras.layers.Convolution2D(32, (3, 3), padding='valid', activation='relu'))
model.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu'))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model with the same parameters
model.compile(
    optimizer=tf.keras.optimizers.SGD(0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model using the training dataset
model.fit(
    x_train, y_train,
    epochs=5
)

# Print the metrics of the model on the test dataset
test_metrics = model.evaluate(x_test, y_test)
print('test metrics={}'.format(test_metrics))

# Save the model
model.save('model_EMNIST_centralized.h5')
