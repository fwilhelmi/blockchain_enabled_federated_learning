import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import random
import time

np.random.seed(1000)

tf.compat.v1.enable_v2_behavior()
tf.compat.v1.graph_util.extract_sub_graph

# 0. PARAMETERS

# NUM_CLIENTS = 200
NUM_CLIENTS_TEST = 50
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_ROUNDS_FL = 200


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


def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])


# 2. LOAD DATA

# Load the mnist dataset
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()


NUM_CLASSES_PER_USER = 10
client_ids_list = []
client_ids_list_ix = []

# Get the clients IDs in str and int forms
for i in range(0, len(emnist_train.client_ids)-1):
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
            elements.append({'label':sample['label'], 'pixels':sample['pixels']})
    # Generate the dataset object for this specific cient
    updated_dataset = tf.data.Dataset.from_generator(
        lambda: elements, {"label": tf.int32, "pixels": tf.float32})
    # Return the dataset
    return updated_dataset


# Generate the new training dataset
pruned_emnist_train = tff.simulation.datasets.ClientData.from_clients_and_fn(
    client_ids_list_ix, create_tf_dataset_for_client_fn)

# Define the ML model compliant with FL (needs a sample of the dataset to be defined)
sample_dataset = pruned_emnist_train.create_tf_dataset_for_client(pruned_emnist_train.client_ids[0])
preprocessed_sample = preprocess(sample_dataset)


def model_fn():  # Model constructor (needed to be passed to TFF, instead of a model instance)
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
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_CLIENT),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_SERVER))

PARTITIONS = [200]
PERCENTAGES = [0.75, 1]

# 3. TRAIN A MODEL
for partition in PARTITIONS:

    print(' + Dataset size: ' + str(partition))
    # Take a subset of the dataset
    subset_ix = np.random.choice(client_ids_list_ix, partition)
    # print(subset_ix)

    for percentage in PERCENTAGES:

        print('     - Training percentage: ' + str(percentage))

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
            # participating_clients = np.random.choice(emnist_train.client_ids, size=NUM_CLIENTS)  # emnist_train.client_ids[0:NUM_CLIENTS-1]
            training_ixs = np.random.choice(subset_ix, round(partition*percentage), replace=False) #emnist_train.client_ids[0:round(partition*percentage)]
            train_datasets = make_federated_data(pruned_emnist_train, training_ixs)
            # print(train_datasets)
            train_time_start = time.time()
            state, train_metrics = iterative_process.next(state, train_datasets)
            # print('  - Round  {}, train metrics={}'.format(round_num, train_metrics))
            train_time_end = time.time()
            training_time.append(train_time_end - train_time_start)
            #  - Get training metrics
            train_loss.append(train_metrics['train']['loss'])
            train_accuracy.append(train_metrics['train']['sparse_categorical_accuracy'])
            #  - Get test metrics
            ixes_original = []
            for i in training_ixs:
                ixes_original.append(emnist_test.client_ids[i])
            test_datasets = make_federated_data(emnist_test, ixes_original)
            test_metrics = fed_evaluation(state.model, test_datasets)
            test_loss.append(test_metrics['loss'])
            test_accuracy.append(test_metrics['sparse_categorical_accuracy'])
            # print('  - Round  {}, test metrics={}'.format(round_num, test_metrics))

            # Choose another set of random clients for evaluation
            sample_random_clients_ids = random.sample(range(0, len(emnist_test.client_ids) - 1), NUM_CLIENTS_TEST)
            sample_random_clients = []
            for idx in sample_random_clients_ids:
                sample_random_clients.append(emnist_test.client_ids[idx])
            eval_datasets = make_federated_data(emnist_test, sample_random_clients)
            eval_metrics = fed_evaluation(state.model, eval_datasets)
            eval_metrics = fed_evaluation(iterative_process.get_model_weights(state), eval_datasets)
            # print('  - Round  {}, eval metrics={}'.format(round_num, eval_metrics))
            eval_loss.append(eval_metrics['loss'])
            eval_accuracy.append(eval_metrics['sparse_categorical_accuracy'])

            round_time_end = time.time()
            iteration_time.append(round_time_end - round_time_start)

        # SAVE RESULTS
        np.savetxt('train_loss_K' + str(partition) + '_' + str(percentage) + '.txt', np.reshape(train_loss, (1, NUM_ROUNDS_FL)))
        np.savetxt('train_accuracy_K' + str(partition) + '_' + str(percentage) + '.txt', np.reshape(train_accuracy, (1, NUM_ROUNDS_FL)))
        np.savetxt('test_loss_K' + str(partition) + '_' + str(percentage) + '.txt', np.reshape(test_loss, (1, NUM_ROUNDS_FL)))
        np.savetxt('test_accuracy_K' + str(partition) + '_' + str(percentage) + '.txt', np.reshape(test_accuracy, (1, NUM_ROUNDS_FL)))
        np.savetxt('eval_loss_K' + str(partition) + '_' + str(percentage) + '.txt', np.reshape(eval_loss, (1, NUM_ROUNDS_FL)))
        np.savetxt('eval_accuracy_K' + str(partition) + '_' + str(percentage) + '.txt', np.reshape(eval_accuracy, (1, NUM_ROUNDS_FL)))
        np.savetxt('iteration_time_K' + str(partition) + '_' + str(percentage) + '.txt', np.reshape(iteration_time, (1, NUM_ROUNDS_FL)))
