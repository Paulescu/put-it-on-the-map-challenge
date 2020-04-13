import math
import os
import pandas as pd
import cv2
import numpy as np


def id2path(images_dir, image_id):
    """Auxiliar function that maps images ids to images full paths"""
    image_file = image_id + '.png'
    return os.path.join(images_dir, image_file)


def path2id(path):
    return os.path.basename(path).split('.')[0]


def id2file(image_id):
    return image_id + '.png'


def get_percentage_pixels_given_color(image, color_rgb):
    """
    Given 'color_rgb' this function returns the percentage of image pixels
    of this color.
    """
    diff = 0
    boundaries = [
        ([color_rgb[2] - diff, color_rgb[1] - diff, color_rgb[0] - diff],
         [color_rgb[2] + diff, color_rgb[1] + diff, color_rgb[0] + diff])]
    # in order BGR as opencv represents images as numpy arrays in reverse order

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(image, lower, upper)
        ratio_color = cv2.countNonZero(mask) / (image.size / 3)

    return ratio_color


def get_image_type(image_path):
    """
    Image type in {'white-orange', 'white-gray', 'blue-green'}
    """
    image = cv2.imread(image_path, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get percentage of white pixels in image
    ratio_white_pixels = get_percentage_pixels_given_color(image,
                                                           [255, 255, 255])

    # get percentage of orange pixels in image
    ratio_gray_pixels = get_percentage_pixels_given_color(image,
                                                          [128, 128, 128])

    if (ratio_white_pixels > 0.20) and (ratio_gray_pixels > 0.20):
        image_type = 'white-gray'
    elif (ratio_white_pixels > 0.20):
        image_type = 'white-orange'
    else:
        image_type = 'blue-green'

    return image_type