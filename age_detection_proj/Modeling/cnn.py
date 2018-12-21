# creating a CNN for audio classification
# the images are log-mel spectrogram

# must write a function to intitialize weights
def init_weights(shape):
    # initialize with a truncated random normal distribution
    init_random_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist)

# must write a function to initialize bias term
def init_bias(shape):
    # just initialize the bias terms to 0.1
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)

# must write a function to perform the 2d convolution
# essentially a wrapper
def conv2d(x,W):
    # x is in the format [batch,H,W,Ch] - this is our input data
    # W is in the format [filter H,filter W,Ch IN,Ch OUT]
    
    # use strides of 1 in every direction
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# write a function for a pooling layer
def max_pool_2by2(x):
    # x is in the format [batch,H,W,Ch]
    
    # k_size is the window size of the pooling
    # i.e. [2,2] takes the max of every 2 by 2 window
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# Conv layer
def convolutional_layer(input_x,shape):
    # initialize weights of layer
    W = init_weights(shape)
    b = init_bias([shape[3]]) # because the output term is represented with
    # fourth element of shape tuple
    ## now put into an activation function
    return tf.nn.relu(conv2d(input_x,W)+b)

# normal layer
def normal_full_layer(input_layer,size):
    # gets the amount of rows in the input layer
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size,size])
    b = init_weights([size])
    return tf.matmul(input_layer,W) + b