def minmaxnormalization(vector):
    """
    Makes the min max normalization over a numpy vector
    """
    vector = vector / (vector.max())
    vector = vector - (vector.min())
    return vector