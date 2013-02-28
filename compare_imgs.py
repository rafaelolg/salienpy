#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
import cv2
import numpy
import sys
from time import time
import salienpy.frequency_tuned
import salienpy.signature
import salienpy.dictionary_frequency
import salienpy.comparison

def main(img_a, img_b):
    saliency_methods = [ ('dictionary_ica_saliency', salienpy.dictionary_frequency.dictionary_saliency),
                       ]

    for name, method in saliency_methods:
        t = time()
        sal_a = method(img_a.copy())
        sal_b = method(img_b.copy())
        corr = salienpy.comparison.compare_saliencies(sal_a,sal_b)
        t = t - time()
        return corr


if __name__ == '__main__':
    if len(sys.argv) > 2:
        img_a  = cv2.imread(sys.argv[1])
        img_b  = cv2.imread(sys.argv[2])
        corr = main(img_a, img_b)
        print '%s <br> ![original](%s) ![protan](%s)'%(corr, sys.argv[1], sys.argv[2])


# compare_imgs.py
