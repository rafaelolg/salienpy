#!/usr/bin/env python
import cv2
import numpy
import sys
import salienpy.frequency_tuned
import salienpy.signature
import salienpy.kmeans_frequency


def main(img):
    cv2.imshow('Original Image', img)
    saliency_methods = [
                ('frequency_tuned',
                    salienpy.frequency_tuned.frequency_tuned_saliency),
                ('signature',
                    salienpy.signature.signature_saliency),
                ('kmeans_frequency',
                        salienpy.kmeans_frequency.kmeans_frequency_saliency)]

    for name, method in saliency_methods:
        sal_img = method(img)
        cv2.imshow(name, sal_img)
        cv2.imwrite(name + '.png', sal_img)
    cv2.waitKey()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img  = cv2.imread(sys.argv[1])
    else:
        cam = cv2.VideoCapture(0)
        status, img = cam.read()
    main(img)

