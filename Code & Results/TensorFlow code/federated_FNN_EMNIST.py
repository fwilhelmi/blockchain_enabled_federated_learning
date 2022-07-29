import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import random
import time
import extra_keras_datasets.emnist as centr_emnist  # See https://github.com/machinecurve/extra_keras_datasets
import nest_asyncio

nest_asyncio.apply()
np.random.seed(1000)

tf.compat.v1.enable_v2_behavior()
tf.compat.v1.graph_util.extract_sub_graph

# 0. PARAMETERS

# NUM_CLIENTS = 200
SELECTED_MODEL = 1  # 1: First FNN, 2: Second FNN, 3: CNN
NUM_CLIENTS_TEST = 50
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 50
PREFETCH_BUFFER = 10
NUM_ROUNDS_FL = 200
AVERAGING_MODEL = 0  # 0: 'fed_avg', 1: 'fed_prox'
NUM_CLASSES_PER_USER = 3

# 1. METHODS

# Pre-processing function
def preprocess(dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


# Make data federated
def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]


# Define an NN model for MNIST
def create_keras_model():
    if SELECTED_MODEL == 1:
        return tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(784)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(10, kernel_initializer='zeros'),
            tf.keras.layers.Softmax(),
        ])
    elif SELECTED_MODEL == 2:
        return tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(784)),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(10, kernel_initializer='zeros'),
            tf.keras.layers.Softmax(),
        ])
    elif SELECTED_MODEL == 3:

        # return tf.keras.models.Sequential([
        #     tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
        #     tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation="relu", padding="same", strides=1),
        #     tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'),
        #     tf.keras.layers.Conv2D(64, kernel_size=(5, 5), activation="relu", padding="same", strides=1),
        #     tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(512, activation="relu"),
        #     tf.keras.layers.Dense(10, activation="softmax"),
        # ])

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,))),
        model.add(tf.keras.layers.Convolution2D(32, (3, 3), padding='valid', activation='relu'))
        model.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu'))

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        return model

#model.add(Conv2D(filters=128, kernel_size=(5,5), padding = 'same', activation='relu',input_shape=(HEIGHT, WIDTH,1)))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
# model.add(Conv2D(filters=64, kernel_size=(3,3) , padding = 'same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(Dropout(.5))
# model.add(Dense(units=num_classes, activation='softmax'))

# 2. LOAD DATA

# Load the mnist dataset
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

# Get the clients IDs in str and int forms
client_ids_list = []
client_ids_list_ix = []
for i in range(0, len(emnist_train.client_ids) - 1):
    client_ids_list.append(emnist_train.client_ids[i])
    client_ids_list_ix.append(i)


def create_tf_dataset_for_client_fn(client_id):
    # Get the original client's dataset to be modified
    client_dataset_copy = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[client_id])
    # Choose random classes to remain
    classes_set = np.random.choice(range(0, 10), NUM_CLASSES_PER_USER, replace=False)
    # List to store the valid samples
    elements = []
    # Iterate for each element in the original client's dataset
    for sample in client_dataset_copy:
        # Select only the samples matching with classes_set
        if sample['label'].numpy() in classes_set:
            elements.append({'label': sample['label'], 'pixels': sample['pixels']})
    # Generate the dataset object for this specific cient
    updated_dataset = tf.data.Dataset.from_generator(
        lambda: elements, {"label": tf.int32, "pixels": tf.float32})
    # Return the dataset
    return updated_dataset


# Generate the new training dataset
pruned_emnist_train = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
    client_ids_list_ix, create_tf_dataset_for_client_fn)

# Define the ML model compliant with FL (needs a sample of the dataset to be defined)
sample_dataset = pruned_emnist_train.create_tf_dataset_for_client(pruned_emnist_train.client_ids[0])
preprocessed_sample = preprocess(sample_dataset)


# Model constructor (needed to be passed to TFF, instead of a model instance)
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_sample.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                 tf.keras.metrics.MeanAbsoluteError()]
    )
    # More loss functions and metrics here:
    # - https://www.tensorflow.org/api_docs/python/tf/keras/losses m
    # - https://www.tensorflow.org/api_docs/python/tf/keras/metrics


fed_evaluation = tff.learning.build_federated_evaluation(model_fn)

# Define the iterative process to be followed for training both clients and the server
# More optimizers here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
LEARNING_RATE_CLIENT = 0.01
LEARNING_RATE_SERVER = 1.00

if AVERAGING_MODEL == 0:
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_CLIENT),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_SERVER))
elif AVERAGING_MODEL == 1:
    iterative_process = tff.learning.algorithms.build_weighted_fed_prox(
        model_fn,
        proximal_strength=0.1,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_CLIENT),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_SERVER))
else:
    print("Unknown averaging model: {}".format(AVERAGING_MODEL))

# Compute the size of the model, based on its parameters
mock_model = create_keras_model()
number_of_model_parameters = mock_model.count_params()
print("Number of parameters: {}".format(number_of_model_parameters))
transaction_size = number_of_model_parameters * 2 / 1000000
print("Model (transaction) size: {}".format(transaction_size))
print(mock_model.summary())

NUM_CLIENTS_PER_ROUND = [200]
PERCENTAGES = [1] #[0.1, 0.25, 0.5, 0.75, 1]

# 3. TRAIN A MODEL
for m in NUM_CLIENTS_PER_ROUND:

    print(' + Number of clients: ' + str(m))
    # Take a subset of the dataset
    subset_ix = np.random.choice(client_ids_list_ix, m)
    # print(subset_ix)

    for percentage in PERCENTAGES:

        print('     - Percentage of clients participating in each round: ' + str(percentage))

        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        eval_loss = []
        eval_accuracy = []
        training_time = []
        iteration_time = []

        # Initialize the server state
        state = iterative_process.initialize()

        # Iterate for each communication round in FL
        for round_num in range(0, NUM_ROUNDS_FL):

            round_time_start = time.time()

            # Training round
            #  - Get a random sample of clients
            training_ixs = np.random.choice(subset_ix, round(m * percentage),
                                            replace=False)  # emnist_train.client_ids[0:round(partition*percentage)]
            train_datasets = make_federated_data(pruned_emnist_train, training_ixs)
            # print(train_datasets)
            train_time_start = time.time()
            result = iterative_process.next(state, train_datasets)
            state = result.state
            train_metrics = result.metrics
            print('round  {}, metrics={}'.format(round_num, train_metrics))

            # print('  - Round  {}, train metrics={}'.format(round_num, train_metrics))
            train_time_end = time.time()
            training_time.append(train_time_end - train_time_start)
            #  - Get training metrics
            clients_metrics = train_metrics['client_work']
            train_loss.append(clients_metrics['train']['loss'])
            #  - Get test metrics for users participating in the current round
            ixes_original = []
            for i in training_ixs:
                ixes_original.append(emnist_test.client_ids[i])
            test_datasets = make_federated_data(emnist_test, ixes_original)
            model_weights = iterative_process.get_model_weights(state)
            test_metrics = fed_evaluation(model_weights, test_datasets)
            test_clients_metrics = test_metrics['eval']
            test_loss.append(test_clients_metrics['loss'])
            test_accuracy.append(test_clients_metrics['sparse_categorical_accuracy'])
            # print('  - Round  {}, test metrics={}'.format(round_num, test_metrics))

            # Choose another set of random clients for evaluation
            sample_random_clients_ids = random.sample(range(0, len(emnist_test.client_ids) - 1), NUM_CLIENTS_TEST)
            sample_random_clients = []
            for idx in sample_random_clients_ids:
                sample_random_clients.append(emnist_test.client_ids[idx])
            eval_datasets = make_federated_data(emnist_test, sample_random_clients)
            eval_metrics = fed_evaluation(model_weights, eval_datasets)
            # print('  - Round  {}, eval metrics={}'.format(round_num, eval_metrics))
            eval_clients_metrics = eval_metrics['eval']
            eval_loss.append(eval_clients_metrics['loss'])
            eval_accuracy.append(eval_clients_metrics['sparse_categorical_accuracy'])

            round_time_end = time.time()
            iteration_time.append(round_time_end - round_time_start)

        # Create a final model and load the last server weights
        final_model = create_keras_model()
        final_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                     tf.keras.metrics.MeanAbsoluteError()]
        )
        model_weights = iterative_process.get_model_weights(state)
        model_weights.assign_weights_to(final_model)
        final_model.save('model_EMNIST_' + str(m) + '_' + str(percentage) + '.h5')  # Save the model

        # SAVE THE RESULTS
        np.savetxt('train_loss_K' + str(m) + '_' + str(percentage) + '.txt',
                   np.reshape(train_loss, (1, NUM_ROUNDS_FL)))
        np.savetxt('test_loss_K' + str(m) + '_' + str(percentage) + '.txt',
                   np.reshape(test_loss, (1, NUM_ROUNDS_FL)))
        np.savetxt('test_accuracy_K' + str(m) + '_' + str(percentage) + '.txt',
                   np.reshape(test_accuracy, (1, NUM_ROUNDS_FL)))
        np.savetxt('eval_loss_K' + str(m) + '_' + str(percentage) + '.txt',
                   np.reshape(eval_loss, (1, NUM_ROUNDS_FL)))
        np.savetxt('eval_accuracy_K' + str(m) + '_' + str(percentage) + '.txt',
                   np.reshape(eval_accuracy, (1, NUM_ROUNDS_FL)))
        np.savetxt('iteration_time_K' + str(m) + '_' + str(percentage) + '.txt',
                   np.reshape(iteration_time, (1, NUM_ROUNDS_FL)))