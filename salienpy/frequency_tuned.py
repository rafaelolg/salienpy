#!/usr/bin/env python
import cv2
import numpy
from commons import minmaxnormalization

def frequency_tuned_saliency(img):
    """
    Frequency Tuned Saliency.
    Find the Euclidean distance between the Lab pixel vector in a Gaussian filtered image
    with the average Lab vector for the input image.
    R. Achanta, S. Hemami, F. Estrada and S. Susstrunk, Frequency-tuned Salient Region
    Detection, IEEE International Conference on Computer Vision and Pattern Recognition

    Args:
        img (numpy.array): an image.

    Returns:
       a 2d image saliency map.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #mean of each channel
    means = []
    for c in range(img.shape[2]):
        means.append(img[:,:,c].mean())
    means = numpy.asarray(means)

    img = cv2.medianBlur(img, 9)
    dist = (img - means)**2
    print("mean color is %s"% means)
    salmap = numpy.zeros((dist.shape[0], dist.shape[1]))
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            salmap[i][j] = numpy.sqrt(dist[i][j].sum())
    return minmaxnormalization(salmap)

