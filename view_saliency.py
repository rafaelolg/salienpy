#!/usr/bin/env python
import cv2
import numpy
import sys
import salienpy.frequency_tuned
import salienpy.signature

def main(img):
    cv2.imshow('Original Image', img)
    ftuned = salienpy.frequency_tuned.frequency_tuned_saliency(img)
    cv2.imshow('Frequency Tuned', ftuned)
    signa = salienpy.signature.signature_saliency(img)
    cv2.imshow('Signature Saliency', signa)
    cv2.waitKey()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        img  = cv2.imread(sys.argv[1])
    else:
        cam = cv2.VideoCapture(0)
        status, img = cam.read()
    main(img)

