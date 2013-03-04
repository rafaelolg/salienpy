#!/usr/bin/env python
# -*- coding: utf-8 -*-
from skimage.transform.pyramids import pyramid_laplacian,resize


def multiscale_saliency(image, method, min_image_area=10000):
    '''
    Runs any saliency method as a multiscale method.
    method is run for each image downsized until its area is lower than min_image_area.
    The final result is an image with the same size as the original.

    '''
    sals = None
    count = 0
    for img in pyramid_laplacian(image, 1):
        print 'calculating for shape = %s, %s'% img.shape[:2]
        s = method(image)
        s = resize(s, image.shape[:2], mode='nearest')
        if sals is not None:
            sals  = sals + s
        else:
            sals = s
        count +=1
    return sals/count


def _ispar_shape(img):
    par = img.shape[0] % 2 == 0 and img.shape[1] == 0
    return par

# multiscale.py
