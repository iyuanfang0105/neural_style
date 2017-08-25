import numpy as np
import scipy.io as sio

import tensorflow as tf
import tools

from matplotlib import pyplot as plt
import math

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)


def load_weights(net_path):
    """

    :param net_path: pretrained vgg19 network stored as mat format
    :return:
    """
    weights = sio.loadmat(net_path)
    if not all(i in weights for i in ('layers', 'classes', 'normalization')):
        raise ValueError("You're using the wrong VGG19 data. Please follow the instructions in the README to download the correct data.")
    mean = weights['normalization'][0][0][0]  # mean image of vgg19 (224, 224, 3)
    mean_pixel = np.mean(mean, axis=(0, 1))  # mean pixels of RGB channels (a, b, c)
    weights = weights['layers'][0]
    return weights, mean_pixel


def net_infer(weights, imput_image, pooling=''):
    net = {}
    current = imput_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernals, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernals = np.transpose(kernals, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernals, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, pooling)
        net[name] = current
    assert len(net) == len(VGG19_LAYERS)
    return net


def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                              padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                              padding='SAME')


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


if __name__ == '__main__':
    pretrained_vgg_19_path = '/home/meizu/WORK/code/neural_style/model/imagenet-vgg-verydeep-19.mat'
    test_image_path = '/home/meizu/WORK/code/neural_style/images/1-content.jpg'
    figures_save_path = '/home/meizu/WORK/code/neural_style/figures'
    # read image
    test_image = tools.imread(test_image_path)

    # load pretrained vgg19 weights
    weights, mean_pixel = load_weights(pretrained_vgg_19_path)

    # preprocess the test image
    test_image_preprocessed = np.array([preprocess(test_image, mean_pixel)]).astype(np.float32)

    # infer the input image with vgg19
    imput_image_infered = net_infer(weights, test_image_preprocessed)

    sess = tf.InteractiveSession()
    aa = sess.run(imput_image_infered['relu4_2'])
    aa = aa.reshape(aa.shape[1], aa.shape[2], aa.shape[3])
    aa_for_plot = []
    for i in range(aa.shape[2]):
        aa_for_plot.append(aa[:,:,i])

    tools.show_images(aa_for_plot[::8], cols=8)
    print ''

