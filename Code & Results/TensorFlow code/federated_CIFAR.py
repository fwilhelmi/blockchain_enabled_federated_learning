import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import random
import time

import nest_asyncio

nest_asyncio.apply()

np.random.seed(1000)

tf.compat.v1.enable_v2_behavior()
tf.compat.v1.graph_util.extract_sub_graph

# gpu_devices = tf.config.list_physical_devices('GPU')
# if not gpu_devices:
#   raise ValueError('Cannot detect physical GPU device in TF')
# tf.config.set_logical_device_configuration(
#     gpu_devices[0],
#     [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#      tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
# tf.config.list_logical_devices()

# 0. PARAMETERS

# NUM_CLIENTS = 200
NUM_CLIENTS_TEST = 50
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10
NUM_ROUNDS_FL = 200
AVERAGING_MODEL = 0  # 0: 'fed_avg', 1: 'fed_prox'


# # Training hyperparameters
# flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
# flags.DEFINE_integer('train_clients_per_round', 4,
#                      'How many clients to sample per round.')
# flags.DEFINE_integer('client_epochs_per_round', 5,
#                      'Number of epochs in the client to take per round.')
# flags.DEFINE_integer('batch_size', 16, 'Batch size used on the client.')
# flags.DEFINE_integer('test_batch_size', 128, 'Minibatch size of test data.')
#
# # Optimizer configuration (this defines one or more flags per optimizer).
# flags.DEFINE_float('server_learning_rate', 1, 'Server learning rate.')
# flags.DEFINE_float('client_learning_rate', 0.0005, 'Client learning rate.')


# 1. METHODS

def element_fn(element):
    return collections.OrderedDict(
        x=tf.expand_dims(element['image'], -1), y=element['label'])


def preprocess_train_dataset(dataset):
    # Use buffer_size same as the maximum client dataset size,
    # 418 for Federated EMNIST
    return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
        count=NUM_EPOCHS)  # .batch(
    # FLAGS.batch_size, drop_remainder=False)


def preprocess_test_dataset(dataset):
    return dataset.map(element_fn).batch(BATCH_SIZE, drop_remainder=False)


# 2. LOAD DATA
#  -> train: 50,000 examples / test: 10,000 examples
train_data, test_data = tff.simulation.datasets.cifar100.load_data()
train_data = train_data.preprocess(preprocess_train_dataset)
# Create a test dataset for all the users
test_data_centralized = preprocess_test_dataset(test_data.create_tf_dataset_from_all_clients())
# Process the test dataset to be federated
test_data = test_data.preprocess(preprocess_train_dataset)
print(test_data)

# Load the CIFAR-100 dataset from tff
# The dataset is derived from the CIFAR-100 dataset. The training and testing examples are partitioned across 500 and 100 clients (respectively).
# No clients share any data samples, so it is a true partition of CIFAR-100. The train clients have string client IDs in the range [0-499],
# while the test clients have string client IDs in the range [0-99]. The train clients form a true partition of the CIFAR-100 training split,
# while the test clients form a true partition of the CIFAR-100 testing split.
# train_data, test_data = tff.simulation.datasets.cifar100.load_data(cache_dir=None)


# Define the number of classes per user (for IIDness purposes)
NUM_CLASSES_PER_USER = 100
client_ids_list = []
client_ids_list_ix = []

# Get the clients IDs in str and int forms
for i in range(0, len(train_data.client_ids) - 1):
    client_ids_list.append(train_data.client_ids[i])
    client_ids_list_ix.append(i)


# Function to manipulate the dataset (to restrict the number of classes per user)
def create_tf_dataset_for_client_fn(client_id):
    # Get the original client's dataset to be modified
    client_dataset_copy = train_data.create_tf_dataset_for_client(
        train_data.client_ids[client_id])
    # Choose random classes to remain
    classes_set = np.random.choice(range(0, 100), NUM_CLASSES_PER_USER, replace=False)
    # List to store the valid samples
    elements = []
    # Iterate for each element in the original client's dataset
    for sample in client_dataset_copy:
        # Select only the samples matching with classes_set
        if sample['y'].numpy() in classes_set:
            elements.append({'y': sample['y'], 'x': sample['x'][:, :, 1]})
    # Generate the dataset object for this specific cient
    updated_dataset = tf.data.Dataset.from_generator(
        lambda: elements, {"y": tf.int32, "x": tf.float32})
    # Return the dataset
    return updated_dataset

# Generate the new training dataset according to the number of available classes per client
pruned_cifar_train = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
    client_ids_list_ix, create_tf_dataset_for_client_fn)

# # Define the ML model compliant with FL (needs a sample of the dataset to be defined)
# # sample_dataset = pruned_cifar_train.create_tf_dataset_for_client(pruned_cifar_train.client_ids[0])
# # preprocessed_sample = preprocess(sample_dataset)
# example_dataset = pruned_cifar_train.create_tf_dataset_for_client(
#     pruned_cifar_train.client_ids[0])
# example_element = next(iter(example_dataset))
# # Plot some examples of images from the first client
# import matplotlib.pyplot as plt
#
# figure = plt.figure(figsize=(20, 4))
# j = 0
# for example in example_dataset.take(40):
#     plt.subplot(4, 10, j + 1)
#     plt.imshow(example['x'].numpy(), cmap='gray', aspect='equal')
#     plt.axis('off')
#     j += 1
# plt.savefig('example.png')


# VGG19 model
def create_vgg19_model():
    model = tf.keras.applications.VGG19(include_top=True,
                                        weights=None,
                                        input_shape=(32, 32, 3),
                                        classes=100)
    return model


# Resnet50 model
def create_resnet_model():
    model = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet',
                                                  input_tensor=tf.keras.layers.Input(shape=(32,
                                                  32, 3)), pooling=None)
    return model


# Model constructor (needed to be passed to TFF, instead of a model instance)
def model_fn():
    keras_model = create_resnet_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=test_data_centralized.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    # More loss functions and metrics here:
    # - https://www.tensorflow.org/api_docs/python/tf/keras/losses m
    # - https://www.tensorflow.org/api_docs/python/tf/keras/metrics


# Define the iterative process to be followed for training both clients and the server
# More optimizers here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# LEARNING_RATE_CLIENT = 0.0005 # 0.0001
# LEARNING_RATE_SERVER = 1.00
# iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
#     model_fn,
#     client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_CLIENT),
#     server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_SERVER))
LEARNING_RATE_CLIENT = 0.02 # 0.0001
LEARNING_RATE_SERVER = 1.00
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_CLIENT),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_SERVER))


# Initialize the state of the FL server
server_state = iterative_process.initialize()
# Generate a federated evaluation object for the model
fed_evaluation = tff.learning.build_federated_evaluation(model_fn)
metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Compute the size of the model, based on its parameters
mock_model = create_resnet_model()  #create_vgg19_model()
number_of_model_parameters = mock_model.count_params()
print("Number of parameters: {}".format(number_of_model_parameters))
transaction_size = number_of_model_parameters * 4 / 1000000
print("Model (transaction) size: {}".format(transaction_size))
print(mock_model.summary())

mock_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# Define the set of experiments to be executed in terms of total number of users and percentages (Async. operation)
NUM_CLIENTS_PER_ROUND = [100] #[10, 50, 100]
PERCENTAGES = [1] #[0.1, 0.25, 0.5, 0.75, 1]

# 3. TRAIN A MODEL
for m in NUM_CLIENTS_PER_ROUND:

    print(' + Number of clients: ' + str(m))

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
            #  - Get a random sample of clients
            sampled_clients = np.random.choice(train_data.client_ids, size=round(m * percentage), replace=False)
            sampled_train_data = [
                train_data.create_tf_dataset_for_client(client).batch(BATCH_SIZE, drop_remainder=False)
                for client in sampled_clients]
            #  - Make an FL iteration
            train_time_start = time.time()
            result = iterative_process.next(server_state, sampled_train_data)
            server_state = result.state
            train_metrics = result.metrics
            print('          * Round  {}, train metrics={}'.format(round_num, train_metrics))
            train_time_end = time.time()
            training_time.append(train_time_end - train_time_start)
            #  - Get training metrics
            clients_metrics = train_metrics['client_work']
            train_loss.append(clients_metrics['train']['loss'])
            train_accuracy.append(clients_metrics['train']['sparse_categorical_accuracy'])
            #  - Get training metrics (TODO)
            # # METHOD 2 (NOT WORKING): USE THE TEST DATASET IN FED_EVALUATION
            test_clients_ids = np.random.choice(test_data.client_ids, size=50, replace=False)
            model_weights = iterative_process.get_model_weights(state)
            sampled_test_data = [
                test_data.create_tf_dataset_for_client(client).batch(BATCH_SIZE, drop_remainder=False)
                for client in test_clients_ids]
            eval_metrics = fed_evaluation(model_weights, sampled_test_data)
            print('          * Round  {}, eval metrics={}'.format(round_num, eval_metrics))
            eval_clients_metrics = eval_metrics['eval']
            eval_loss.append(eval_clients_metrics['loss'])
            eval_accuracy.append(eval_clients_metrics['sparse_categorical_accuracy'])
            # # WORKAROUND 1: USE RANDOM OTHER SAMPLES IN THE TRAINING DATASET TO USE FED_EVALUATION
            # test_clients_ids = np.random.choice(train_data.client_ids, size=10, replace=False)
            # model_weights = iterative_process.get_model_weights(state)
            # sampled_test_data = [
            #     train_data.create_tf_dataset_for_client(client).batch(BATCH_SIZE, drop_remainder=False)
            #     for client in test_clients_ids]
            # eval_metrics = fed_evaluation(model_weights, sampled_test_data)
            # print(eval_metrics)
            # eval_clients_metrics = eval_metrics['eval']
            # eval_loss.append(eval_clients_metrics['loss'])
            # eval_accuracy.append(eval_clients_metrics['sparse_categorical_accuracy'])
            # # WORKAROUND 2 (NOT WORKING): USE THE ENTIRE TEST DATASET FOR EVALUATION OF THE MODEL
            # model_weights.assign_weights_to(mock_model)
            # eval_metrics = mock_model.evaluate(test_data)
            # # eval_metrics = fed_evaluation(model_weights, test_data[0])
            # # print('  - Round  {}, eval metrics={}'.format(round_num, eval_metrics))
            # print('  - Round  {}, test metrics={}'.format(round_num, eval_metrics))
            round_time_end = time.time()
            iteration_time.append(round_time_end - round_time_start)

        # Create a final model and load the last server weights
        final_model = create_vgg19_model()
        final_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        model_weights = iterative_process.get_model_weights(state)
        model_weights.assign_weights_to(final_model)
        final_model.save('model_CIFAR_' + str(m) + '_' + str(percentage) + '.h5')  # Save the model

        # SAVE RESULTS
        np.savetxt('output_cifar/train_loss_K' + str(m) + '_' + str(percentage) + '.txt',
                   np.reshape(train_loss, (1, NUM_ROUNDS_FL)))
        np.savetxt('output_cifar/train_accuracy_K' + str(m) + '_' + str(percentage) + '.txt',
                   np.reshape(train_accuracy, (1, NUM_ROUNDS_FL)))
        np.savetxt('output_cifar/eval_loss_K' + str(m) + '_' + str(percentage) + '.txt',
                   np.reshape(eval_loss, (1, NUM_ROUNDS_FL)))
        np.savetxt('output_cifar/eval_accuracy_K' + str(m) + '_' + str(percentage) + '.txt',
                   np.reshape(eval_accuracy, (1, NUM_ROUNDS_FL)))
        np.savetxt('output_cifar/iteration_time_K' + str(m) + '_' + str(percentage) + '.txt',
                   np.reshape(iteration_time, (1, NUM_ROUNDS_FL)))
