#coding=utf-8
import os
import struct
import tensorflow as tf 
import numpy as np 
import pdb
from datetime import datetime
from VGG16 import *

batch_size = 64
lr = 0.0001
n_cls = 10
max_steps = 1000

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    images = images.reshape(len(images),28,28,1)
    images = images / 255.0
    labels = tf.one_hot(labels, n_cls, 1, 0)
    return images, labels

def train():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name='input')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_cls], name='label')
    keep_prob = tf.placeholder(tf.float32)
    output = VGG16(x, keep_prob, n_cls)
    #probs = tf.nn.softmax(output)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output,1), tf.argmax(y, 1)), tf.float32))
    
#    images, labels = read_and_decode('./train.tfrecords')
    images, labels = load_mnist(r'.\fashion', kind='train')
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=False)
    img_batch, label_batch = tf.train.batch(input_queue,
                                            batch_size=batch_size,
                                            capacity=392)
#    label_batch = tf.one_hot(label_batch, n_cls, 1, 0)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(21)
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #saver.restore(sess,'./model/model.ckpt-999')
        for i in range(max_steps * 20):
            batch_x, batch_y = sess.run([img_batch, label_batch])
#            print batch_x, batch_x.shape
#            print batch_y
#            pdb.set_trace()
            _, loss_val = sess.run([train_step, loss], feed_dict={x:batch_x, y:batch_y, keep_prob:0.5})
            if i%10 == 0:
                train_arr = accuracy.eval(feed_dict={x:batch_x, y: batch_y, keep_prob: 1.0})
                print("%s: Step [%d]  Loss : %f, training accuracy :  %g" % (datetime.now(), i, loss_val, train_arr))
            if (i + 1) == max_steps:
                #checkpoint_path = os.path.join(FLAGS.train_dir, './model/model.ckpt')
                saver.save(sess, './model/model.ckpt', global_step=i+1)
        coord.request_stop()
        coord.join(threads)
        #saver.save(sess, 'model/model.ckpt')


if __name__ == '__main__':
    train()
