#!/usr/bin/env python
import cv2
import numpy
import sys
import salienpy.frequency_tuned
import salienpy.signature
import salienpy.kmeans_frequency
import salienpy.aws


def main(img):
    saliency_methods = [
                ('frequency_tuned',
                    salienpy.frequency_tuned.frequency_tuned_saliency),
                ('aws',
                    salienpy.aws.aws_saliency),
                ('signature',
                    salienpy.signature.signature_saliency),
                ('kmeans_frequency',
                        salienpy.kmeans_frequency.kmeans_frequency_tuned_saliency)]

    for name, method in saliency_methods:
        print name
        sal_img = method(img.copy())
        cv2.imwrite(name + '.png', 255 * sal_img.astype('uint8'))
    cv2.waitKey()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img  = cv2.imread(sys.argv[1])
    else:
        cam = cv2.VideoCapture(0)
        status, img = cam.read()
    main(img)

