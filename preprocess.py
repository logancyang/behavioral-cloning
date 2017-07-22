import numpy as np
import scipy.misc
from scipy.stats import bernoulli


def crop(image, top_percent, bottom_percent):
    assert 0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    assert 0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]


def resize(image, new_dimension):
    return scipy.misc.imresize(image, new_dimension)


def random_flip(image, steering_angle, flipping_prob=0.5):
    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle


def generate_new_image(
        image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
        resize_dimension=(64, 64), generate_new_images=True):
    if generate_new_images:
        image = crop(image, top_crop_percent, bottom_crop_percent)
        image, steering_angle = random_flip(image, steering_angle)
    image = resize(image, resize_dimension)
    return image, steering_angle
