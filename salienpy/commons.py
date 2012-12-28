def minmaxnormalization(vector):
    """
    Makes the min max normalization over a numpy vector
    $$ v_i = (v_i / max(v)) - (min((v_i)/max(v_i))
    """
    vector = vector / (vector.max())
    vector = vector - (vector.min())
    return vector