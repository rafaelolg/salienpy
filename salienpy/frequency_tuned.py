#!/usr/bin/env python
import cv2
import numpy
from commons import minmaxnormalization


def frequency_tuned_saliency(image):
    """
    Frequency Tuned Saliency.
    Find the Euclidean distance between
    the Lab pixel vector in a Gaussian filtered image
    with the average Lab vector for the input image.
    R. Achanta, S. Hemami, F. Estrada and S. Susstrunk,
    Frequency-tuned Salient Region
    Detection, IEEE International Conference on Computer
     Vision and Pattern Recognition

    Args:
        image (numpy.array): an image.

    Returns:
       a 2d image saliency map.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #mean of each channel
    means = []
    for c in range(image.shape[2]):
        means.append(image[:, :, c].mean())
    means = numpy.asarray(means)

    image = cv2.medianBlur(image, 9)
    dist = (image - means) ** 2
    print("mean color is %s" % means)
    salmap = numpy.zeros((dist.shape[0], dist.shape[1]))
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            salmap[i][j] = numpy.sqrt(dist[i][j].sum())
    return minmaxnormalization(salmap)

if __name__ == '__main__':
    import sys
    import re
    import os
    img = cv2.imread(sys.argv[1])
    basename = os.path.basename(sys.argv[1])
    m = re.search('(.*)\..+', basename)
    stem = m.groups()[0]
    sal = frequency_tuned_saliency(img) * 255
    cv2.imwrite("%s.png" % stem, sal)
