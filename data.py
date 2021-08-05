import os
import struct
import numpy as np
import tensorflow as tf

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
    labels = tf.one_hot(labels, 10, 1, 0)
    return images, labels

X_train, y_train = load_mnist(r'.\fashion',kind='train')
X_test, y_test = load_mnist(r'fashion',kind='t10k')


print(X_train.shape)
print(X_train[0])

sess = tf.Session()
print(sess.run(y_train[0:10]))
print(y_train.shape,y_train[0:10])
