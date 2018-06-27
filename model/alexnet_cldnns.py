# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a model definition for AlexNet.
This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton
and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014
Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.
Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)
@@alexnet_v2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
import sys
sys.path.append('../data_handle/')
from record_decode import get_batch
from record_decode import decode_from_tfrecords

trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)

lr = 0.001
epoch = 1000
batch_size = 1
n_input = 16
n_step = 16
n_hidden_units = 128
n_class = 2

def debug_shape(t):
    print(type(t), t.get_shape().as_list())

xs = tf.placeholder(shape=[None, 6400], dtype=tf.float32)
ys = tf.placeholder(shape=[None, 2], dtype=tf.int64)



def alexnet_v2(
        inputs,
        num_classes=2,
        is_training=True,
        dropout_keep_prob=0.5,
        spatial_squeeze=True,
        scope='alexnet_v2'):
    inputs = tf.reshape(inputs, (-1, 80, 80, 1))
    with variable_scope.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        debug_shape(inputs)
        with arg_scope(
                [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
                outputs_collections=[end_points_collection]):
            net = layers.conv2d(
                inputs, 64, [3, 3], padding='VALID', scope='conv1')
            debug_shape(net)
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool1')
            debug_shape(net)
            net = layers.conv2d(net, 192, [5, 5], scope='conv2')
            debug_shape(net)
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')
            debug_shape(net)
            net = layers.conv2d(net, 384, [3, 3], scope='conv3')
            net = layers.conv2d(net, 384, [3, 3], scope='conv4')
            net = layers.conv2d(net, 256, [3, 3], scope='conv5')
            net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool5')
            debug_shape(net)
            with arg_scope(
                    [layers.conv2d],
                    weights_initializer=trunc_normal(0.005),
                    biases_initializer=init_ops.constant_initializer(0.1)):
                net = layers.conv2d(net, 256, [7, 7], 2,padding='VALID', scope='fc6')
                debug_shape(net)
                net = layers_lib.dropout(
                net, dropout_keep_prob, is_training=is_training, scope='dropout6')
                net = layers.conv2d(net, 256, [1, 1], scope='fc7')
                debug_shape(net)
            end_points = utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = array_ops.squeeze(net, [1, 2], name='fc8/squeezed')
            end_points[sc.name + '/fc8'] = net
            return net, end_points


# spec, label = decode_from_tfrecords(['../data/p_g/train.tfrecords'])
# batch_spec, batch_label = get_batch(spec, label, batch_size=10)
# batch_spec = tf.reshape(batch_spec, (-1, 80, 80, 1))
# net, _ = alexnet_v2(spec)
# print(debug_shape(batch_spec))
# # alexnet_v2.default_image_size = 224

# weights ans biases
weight = {
    'in': tf.Variable(tf.random_normal([n_input, n_hidden_units])),
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_class]))
}
bias = {
    'in': tf.constant(0.1, shape=[n_hidden_units,]),
    'out': tf.constant(0.1, shape=[n_class,])
}


# RNN
def rnn(X, Weights, biases):
    X = tf.reshape(X, [-1, n_input])
    X_in = tf.matmul(X, weight['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_step, n_hidden_units])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    results = tf.matmul(states[1], weight['out']) + biases['out']
    return results



net, _ = alexnet_v2(inputs=xs)
debug_shape(net)
pred = rnn(net, weight, bias)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=ys))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    spec, label = decode_from_tfrecords(['../data/p_g/train.tfrecords'])
    batch_spec, batch_label = get_batch(spec, label, batch_size=1)
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(epoch):
        batch_xs, batch_ys = sess.run([batch_spec, batch_label])
        batch_xs = batch_xs.astype(np.float32)
        # batch_xs = batch_xs.reshape([batch_size, n_step, n_input])
        sess.run([train_op], feed_dict={
            xs: batch_xs,
            ys: batch_ys,
        })
        if i % 5 == 0:
            # batch_Xval, batch_Yval = mnist.test.next_batch(batch_size)
        #     print(sess.run(accuracy, feed_dict={
        #     xs: batch_xs,
        #     ys: batch_ys,
        # }))
            print(sess.run([loss], feed_dict={
            xs: batch_xs,
            ys: batch_ys,
        }))
            pass
    coord.request_stop()
    coord.join(threads)