def minmaxnormalization(vector):
    """
    Makes the min max normalization over a numpy vector
    $$ v_i = (v_i - min(v)) / max(v)
    """
    vector = vector - (vector.min())
    vector = vector / (vector.max())
    return vector



def gen_even_slices(n, n_packs):
    """Generator to create n_packs slices going up to n.
    Taken from sklearn.utils import gen_even_slices
    Examples
    --------
    >>> list(gen_even_slices(10, 1))
    [slice(0, 10, None)]
    >>> list(gen_even_slices(10, 10))                     #doctest: +ELLIPSIS
    [slice(0, 1, None), slice(1, 2, None), ..., slice(9, 10, None)]
    >>> list(gen_even_slices(10, 5))                      #doctest: +ELLIPSIS
    [slice(0, 2, None), slice(2, 4, None), ..., slice(8, 10, None)]
    >>> list(gen_even_slices(10, 3))
    [slice(0, 4, None), slice(4, 7, None), slice(7, 10, None)]
    """
    start = 0
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            yield slice(start, end, None)
            start = end
