from sys import stderr
import math

import scipy.misc as misc
import numpy as np
import tensorflow as tf
from PIL import Image

import vgg
import tools

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def model_neural_style(pre_train_vgg_path, content_image, style_images,
                       content_weight=5e0, content_weight_blend=1.0,
                       style_weight=5e2, style_layer_weight_exp=1.0,
                       pooling='', initial=None, initial_noiseblend=1.0,
                       tv_weight=1e2, learning_rate=1e1, beta1=0.9,
                       beta2=0.999, epsilon=1e-08, print_iterations=None,
                       iterations=500, checkpoint_iterations=50,
                       preserve_colors=None):
    print "++++++++++++++++++++"
    # input shape of model
    shape = (1,) + content_image.shape
    style_images_shapes = [(1,) + style_image.shape for style_image in style_images]
    content_features = {}
    style_features = [{} for _ in style_images]

    # load the weights of pretrained vgg model
    vgg_weights, vgg_mean_pixel = vgg.load_weights(pre_train_vgg_path)

    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp

    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_infer(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content_image, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})
    # # for debug
    # for layer in CONTENT_LAYERS:
    #     item = content_features[layer]
    #     item = item.reshape(item.shape[1], item.shape[2], item.shape[3])
    #     item_for_plot = []
    #     for i in range(item.shape[2]):
    #         item_for_plot.append(item[:, :, i])
    #
    #     tools.show_images(item_for_plot[::8], cols=8)
            # compute style features in feedforward mode

    # compute styles features in feedforward mode
    for i in range(len(style_images)):
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_images_shapes[i])
            net = vgg.net_infer(vgg_weights, image, pooling)
            style_pre = np.array([vgg.preprocess(style_images[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(features.T, features) / features.size
                style_features[i][layer] = gram

    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content_image) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
            noise = np.random.normal(size=shape, scale=np.std(content_image) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)
        image = tf.Variable(initial)
        net = vgg.net_infer(vgg_weights, image, pooling)

        # content loss
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend
        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
                net[content_layer] - content_features[content_layer]) /
                content_features[content_layer].size))
            content_loss += reduce(tf.add, content_losses)
        # style loss
        style_loss = 0
        for i in range(len(style_images)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / size
                style_gram = style_features[i][style_layer]
                style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
        # total variation denoising
        tv_y_size = _tensor_size(image[:, 1:, :, :])
        tv_x_size = _tensor_size(image[:, :, 1:, :])
        tv_loss = tv_weight * 2 * (
            (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - 1, :, :]) /
             tv_y_size) +
            (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - 1, :]) /
             tv_x_size))

        # overall loss
        loss = content_loss + style_loss + tv_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('       tv loss: %g\n' % tv_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            if (print_iterations and print_iterations != 0):
                print_progress()
            for i in range(iterations):
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()
                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                    if preserve_colors and preserve_colors == True:
                        original_image = np.clip(content_image, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB

                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(
                            Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))

                    yield (
                        (None if last_step else i),
                        img_out
                    )


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

if __name__ == '__main__':
    # prepare the content image for style transform
    # and the specified style image
    content_image_path = '../images/1-content.jpg'
    style_image_path = '../images/1-style.jpg'
    content_image = tools.imread(content_image_path)
    style_image = tools.imread(style_image_path)
    style_images = [style_image]  # for multi-styles
    tools.show_images([content_image, style_image], titles=['content image', 'style'], axis='on')

    # pretrained vgg path
    pre_trained_vgg_path = '../model/imagenet-vgg-verydeep-19.mat'

    # set content image width for model input
    # and style_scale for scaling the style image
    width = 384
    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        content_image = misc.imresize(content_image, new_shape).astype(np.float32)
    target_shape = content_image.shape
    tools.show_images([content_image, style_image], titles=['content image(width=' + str(width) + ')', 'style'], axis='on')

    style_scale = 1.0
    style_scales = [style_scale]  # for multi-style-scales
    for i in range(len(style_images)):
        if style_scales is not None:
            style_scale = style_scales[i]
        style_images[i] = misc.imresize(style_images[i], style_scale * target_shape[1]
                                        / style_images[i].shape[1]).astype(np.float32)
    tools.show_images([content_image, style_images[0]], titles=['content image(width=' + str(content_image.shape[1]) + ')', 'style(width=' + str(style_images[0].shape[1]) + ')'],
                      axis='on')

    # blend style image
    style_blend_weights = None
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0 / len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight / total_blend_weight
                               for weight in style_blend_weights]
    # initial noise blend
    initial_noiseblend = 1.0
    # style layer weight exp
    style_layer_weight_exp = 1.0

    styled_images = []
    for iteration, image in model_neural_style(pre_trained_vgg_path, content_image, style_images):
        output_file = None
        styled_images.append(image.astype(np.float32))

    tools.show_images(styled_images, cols=4)
