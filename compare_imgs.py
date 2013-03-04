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
from salienpy.commons import minmaxnormalization
def main(img_a, img_b):
    saliency_methods = [ ('dictionary_ica_saliency', salienpy.dictionary_frequency.dictionary_saliency),
                       ]

    for name, method in saliency_methods:
        t = time()
        sal_a = method(img_a.copy())
        sal_b = method(img_b.copy())
        t = t - time()
        print 'time = %s'

        return sal_a , sal_b


if __name__ == '__main__':
    if len(sys.argv) > 5:
        img_a  = cv2.imread(sys.argv[1])
        img_b  = cv2.imread(sys.argv[2])
        sal_a, sal_b = main(img_a, img_b)
        cv2.imwrite(sys.argv[3],(255 * minmaxnormalization(sal_a)).astype('uint8'))
        cv2.imwrite(sys.argv[4],(255 * minmaxnormalization(sal_b)).astype('uint8'))
        cv2.imwrite(sys.argv[5],(255 * minmaxnormalization(sal_a - sal_b)).astype('uint8'))
        print '%s\t%s\t%s\t%s\t%s' % sys.argv[1:6]
    else:
        import pprint
        pprint.pprint(sys.argv)

# compare_imgs.py
