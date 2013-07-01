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
from pylab import *
import matplotlib.pyplot as plt
from matplotlib import cm

def main(img_a, img_b):
    saliency_methods = [ ('dictionary_ica_saliency', salienpy.dictionary_frequency.dictionary_saliency),
                       ]

    for name, method in saliency_methods:
        t = time()
        sal_a = method(img_a.copy(), 'kmeans')
        sal_b = method(img_b.copy(), 'kmeans')
        t = t - time()
        print 'time = %s'% t
        print 'correlacao = %s' % salienpy.comparison.compare_saliencies(sal_a, sal_b)
        return sal_a , sal_b


def plot_dif(dif):
        vmax=max([abs(i) for i in (dif.min(), dif.max())])
        dif[0][0] = 3.7
        dif[1][0]= -3.7
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(dif, cmap=cm.bwr)
        cbar= fig.colorbar(cax)
        plt.savefig(sys.argv[5])




if __name__ == '__main__':
    if len(sys.argv) > 5:
        img_a  = cv2.imread(sys.argv[1])
        img_b  = cv2.imread(sys.argv[2])
        sal_a, sal_b = main(img_a, img_b)
        cv2.imwrite(sys.argv[3],(255 * minmaxnormalization(sal_a)).astype('uint8'))
        cv2.imwrite(sys.argv[4],(255 * minmaxnormalization(sal_b)).astype('uint8'))
        dif = sal_a - sal_b
        dif_abs =  numpy.abs(dif)
        #cv2.imwrite('dif_sal_absolut.png',
        #(255 * minmaxnormalization(dif_abs)).astype('uint8'))
        plot_dif(dif)
    else:
        import pprint
        pprint.pprint(sys.argv)

# compare_imgs.py
