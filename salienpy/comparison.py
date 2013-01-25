#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

# comparison.py


def compare_saliencies(first_s, second_s):
    """
    Compare two salieancies using the Pearson Correlation.
    """
    x = first_s.reshape(-1)
    y = second_s.reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    r = ((x*y).sum())/math.sqrt((x*x).sum() * (y*y).sum())
    return r
