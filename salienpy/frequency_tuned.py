#!/usr/bin/env python
import cv2
import numpy

def frequency_tuned_saliency(img):
    """
    Frequency Tuned Saliency.
    Find the Euclidean distance between the Lab pixel vector in a Gaussian filtered image
    with the average Lab vector for the input image.
    R. Achanta, S. Hemami, F. Estrada and S. Süsstrunk, Frequency-tuned Salient Region
    Detection, IEEE International Conference on Computer Vision and Pattern Recognition

    Args:
        img (numpy.array): a 3-channel image color image.

    Returns:
       a 2d image saliency map.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #mean of each channel
    m = numpy.asarray([img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()])
    img = cv2.medianBlur(img, 9)
    dist = (img - m)**2
    print("mean color is %s"% m)
    salmap = numpy.zeros((dist.shape[0], dist.shape[1]))
    for l in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            salmap[l][c] = numpy.sqrt(dist[l][c].sum())
    #minmax normalization
    salmap = (salmap -salmap.min())
    salmap = salmap/salmap.max()
    salmap = 255*salmap

