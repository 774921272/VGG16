#coding=utf-8

import tensorflow as tf
import numpy as np 

data_dict = np.load('./vgg16.npy', encoding='latin1').item()

def print_layer(t):
    print(t.op.name, ' ', t.get_shape().as_list(), '\n')

def conv(x, d_out, name, fineturn=False, xavier=False):
    d_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        # Fine-tuning 
        if fineturn:
            '''
            kernel = tf.Variable(tf.constant(data_dict[name][0]), name="weights")
            bias = tf.Variable(tf.constant(data_dict[name][1]), name="bias")
            '''
            kernel = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print("fineturn")
        elif not xavier:
            kernel = tf.Variable(tf.truncated_normal([3, 3, d_in, d_out], stddev=0.1), name='weights')
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),
                                                trainable=True, 
                                                name='bias')
            print("truncated_normal")
        else:
            kernel = tf.get_variable(scope+'weights', shape=[3, 3, d_in, d_out], 
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),
                                                trainable=True, 
                                                name='bias')
            print("xavier")
        conv = tf.nn.conv2d(x, kernel,[1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        print_layer(activation)
        return activation

def maxpool(x, name):
    activation = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=name) 
    print_layer(activation)
    return activation

def fc(x, n_out, name, fineturn=False, xavier=False):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if fineturn:
            '''
            weight = tf.Variable(tf.constant(data_dict[name][0]), name="weights")
            bias = tf.Variable(tf.constant(data_dict[name][1]), name="bias")
            '''
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print("fineturn")
        elif not xavier:
            weight = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01), name='weights')
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), 
                                                trainable=True, 
                                                name='bias')
            print("truncated_normal")
        else:
            weight = tf.get_variable(scope+'weights', shape=[n_in, n_out], 
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), 
                                                trainable=True, 
                                                name='bias')
            print("xavier")
        # ????????????????????????relu_layer?????????????????????????????????????????????relu??????
        activation = tf.nn.relu_layer(x, weight, bias, name=name)
        print_layer(activation)
        return activation

def VGG16(images, _dropout, n_cls):

    conv1_1 = conv(images, 64, 'conv1_1', xavier=True)
    conv1_2 = conv(conv1_1, 64, 'conv1_2', xavier=True)
    pool1   = maxpool(conv1_2, 'pool1')

    conv2_1 = conv(pool1, 128, 'conv2_1', xavier=True)
    conv2_2 = conv(conv2_1, 128, 'conv2_2', xavier=True)
    pool2   = maxpool(conv2_2, 'pool2')

    conv3_1 = conv(pool2, 256, 'conv3_1', xavier=True)
    conv3_2 = conv(conv3_1, 256, 'conv3_2', xavier=True)
    conv3_3 = conv(conv3_2, 256, 'conv3_3', xavier=True)
#    pool3   = maxpool(conv3_3, 'pool3')

    '''
    ???????????????????????????????????????????????????????????????????????????
    '''
    flatten  = tf.reshape(conv3_3, [-1, 7*7*256])
    fc6      = fc(flatten, 4096, 'fc6', xavier=True)
    dropout1 = tf.nn.dropout(fc6, _dropout)

    fc7      = fc(dropout1, 4096, 'fc7', xavier=True)
    dropout2 = tf.nn.dropout(fc7, _dropout)
    
    fc8      = fc(dropout2, n_cls, 'fc8', xavier=True)

    prob = tf.nn.softmax(fc8, name="prob")

    return prob

