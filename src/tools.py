import numpy as np
from scipy import misc
from PIL import Image
from matplotlib import pyplot as plt


def imread(path):
    """
    read image
    :param path: image path
    :return: RGB image ndarray (h, w, 3)
    """
    img = misc.imread(path).astype(np.float32)
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def imsave(path, img):
    """
    save image
    :param path: image path saved
    :param img: image data (ndarray)
    :return: null
    """
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


def imshow(img):
    """
    show image
    :param img: image data (ndarray)
    :return: null
    """
    misc.imshow(img)


def show_images(images, cols=1, titles=None, axis='off'):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    # if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    if titles is None: titles = ['%d' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        if image.max() > 1.0:
            temp = image / 255.0
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        a.axis(axis)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(temp)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


if __name__ == '__main__':
    test_image_path = '/home/meizu/WORK/code/neural_style/images/1-content.jpg'
    image = imread(test_image_path)
    misc.imshow(image)
    print ''