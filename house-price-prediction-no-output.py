# prediction of house prices from pluralsight course "Tensorflow: Getting Started"

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generate some random house sizes between 1000 and 3500 ft^2
num_house = 100
np.random.seed(100)

# generate house prices from house size with a random offset
house_size = np.random.randint(low=1000, high=3500, size=num_house)
house_price = house_size * 400.0 + np.random.randint(low=-50000, high=100000, size=num_house)

# plot generated house and size
plt.plot(house_size, house_price, "bx") # bx = blue x
plt.ylabel("Price")
plt.xlabel("Size")
# plt.show()

# normalize values to prevent under/overflows
def normalize(array):
    return (array - array.mean()) / array.std()

# define the number of training samples
num_train_samples = math.floor(num_house * .7)

# define training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define test data
test_house_size = np.asarray(house_size[num_train_samples:])
test_price = np.asarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_price_norm = normalize(test_price)

# set up tensorflow placeholders that get updated as we descend down the gradient
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="house_price")

print(house_size)
print(house_price)

# define the variables holding the size_factor and price we set during the training
# initialize them based on some random values based on the normal distrubution
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

print(size_factor)
print(price_offset)

# define the operations for the predicting values
# note use of tensorflow methods
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# define the loss function (how much error) - mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2)) / (2 * num_train_samples)

# optimizer learning rate (the size of the steps down the gradient)
learning_rate = 0.1

# define a gradient descent optimizer that will minimize the loss defined in the operation "cost"
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# initialize the variables
init = tf.global_variables_initializer()

# launch the graph in the session
with tf.Session() as sess:
    sess.run(init)

    # set how often to display training progress and number of training iterations
    display_every = 2
    num_training_iter = 50

    # keep iterating over the training data
    for iteration in range(num_training_iter):

        # fit all training data
        for (x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

        # display current status
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
            print("iteration #:", "%04d" % (iteration + 1), "cost=", "{:.9f}".format(c), \
                "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    print("Optimization finished!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), "\n")

    
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    # plot the graph
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure=()
    plt.ylabel("Price")
    plt.xlabel("House Size (ft^2")
    plt.plot(train_house_size, train_price, 'go', label='Training Data')
    plt.plot(test_house_size, test_price, 'mo', label='Testing Data')
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
        (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
        label='Learned Regression')

    plt.legend(loc='upper left')
    plt.show()