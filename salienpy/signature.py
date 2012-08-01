#!/usr/bin/env python2
import cv2
import numpy as np

def signature_saliency(img):
    """
    Signature Saliency.


    X. Hou, J. Harel, and C. Koch, "Image Signature: Highlighting Sparse Salient
    Regions." IEEE Trans. Pattern Anal. Mach. Intell. 34(1): 194-201 (2012)
    """
    old_shape = (img.shape[0],img.shape[1])
    img = img_padded_for_dct(img)
    img = img/255.0
    sal = []
    for c in range(img.shape[2]):
        channel = img[:,:,c].astype(np.dtype('float32'))
        channel_dct = np.sign(cv2.dct(channel))
        s = cv2.idct(channel_dct)
        s = np.multiply(s,s)
        sal.append(s)
    sal = sum(sal)/3.0
    sal = cv2.GaussianBlur(sal, (11,11), 0)
    sal = sal[:old_shape[0], :old_shape[1]]
    sal = sal / (sal.max())
    sal = sal - (sal.min())
    return sal

def img_padded_for_dct(img):
    h = img.shape[0]
    w = img.shape[1]
    if (h%2 == 1):
        h=h+1
    if (w%2 == 1):
        w=w+1
    return cv2.copyMakeBorder(img, top=0,  bottom=(h-img.shape[0]),
                                   left=0, right=(w-img.shape[1]),
                                   borderType=cv2.BORDER_REPLICATE)
