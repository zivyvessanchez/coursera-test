import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# Convolutional layer 1
filter_size1 = 5 # 5 x 5 filter size
num_filters1 = 16 # Number of filters

# Convolutional layer 2
filter_size2 = 5 # 5 x 5 filter size
num_filters2 = 36 # Number of filters

# Fully connected layer
fc_size = 128 # Number of neurons in FCL

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)

# MNIST image size in pixels (width/height)
img_size = 28

# MNIST image flat size (1D vector)
img_size_flat = img_size * img_size

# MNIST image square size (width x height)
img_shape = (img_size, img_size)

# Number of color channels for images
num_channels = 1

# Number of classification classes. Represents MNIST 0-9
num_classes = 10

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Show plot
    plt.show()

images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images=images, cls_true=cls_true)

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,               # The previous layer
                   num_input_channels,  # Num. channels in prev. layer
                   filter_size,         # Width and height of each filter
                   num_filters,         # Number of filters
                   use_pooling=True):   # Use 2x2 max-pooling

    # Shape of filter-weights for the convolution
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights/filters with given shape
    weights = new_weights(shape=shape)

    # Create new biases, one for each fliter
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         stride=[1,1,1,1]
                         padding='SAME')

    # Add biases to the convolution, to each filter-channel.
    layer += biases
