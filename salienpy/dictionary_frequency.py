import numpy
from sklearn.feature_extraction.image import extract_patches_2d

from sklearn.decomposition import FastICA
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import SparseCoder
from commons import gen_even_slices
from commons import minmaxnormalization





def dictionary_saliency(image,algorithm='ica', show_info=False):
    """
    Dictionary frequency Tuned Saliency.
    Args:
        image (numpy.array): an image.
        algorithm: the algorithm to create the sparse dictionary .
            Accepted values  are kmeans and ica.
    Returns:
       a 2d image saliency map.
    """
    from time import time
    t0 = time()
    components, mean_value = extract_components(image, algorithm)
    train_time = time()
    encoded = image_to_components_space(image, components, mean_value)
    ecoding_time = time()
    encoded = encoded - encoded.mean(axis=0)
    encoded = encoded / abs(encoded).max()
    sal = numpy.sqrt(numpy.sum(encoded * encoded, axis=1))
    sal = numpy.reshape(sal, (image.shape[0] - 11, image.shape[1] - 11))
    frequency_diff_time = time()
    total_time = time()
    if show_info:
        print '(train_time:%s, ecoding_time:%s, frequency_diff_time:%s, total_time: %s)' % (
                train_time - t0,
                ecoding_time - train_time,
                frequency_diff_time - ecoding_time,
                total_time-t0)
        plot_components(components)
    return minmaxnormalization(sal)



def calculate_max_number_of_patches(image, patches_size=(12,12), max_number = 100000):
    fitting_patches = image.shape[0] * image.shape[1]
    fitting_patches = fitting_patches - (image.shape[0]*(patches_size[1] -1))
    fitting_patches = fitting_patches - (image.shape[1]*(patches_size[0] -1))
    return min(fitting_patches, max_number)


def extract_components(image,
                       algorithm,
                       patches_size=(12,12),
                       projection_dimensios=12,
                       previous_components=None):
    """
    Gets a higher dimension sparse dictionary.algorithm can be
    ica, kmeans or colors.

    """
    patches = extract_patches_2d(image, patches_size, calculate_max_number_of_patches(image, patches_size))
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    mean_value  = patches.mean(axis=0)
    patches = (patches - mean_value)
    if algorithm == 'ica':
        ica = FastICA(n_components=projection_dimensios, whiten=True, max_iter=10)
        ica.fit(patches.T)
        components = ica.components_.T
    elif algorithm == 'kmeans':
        kmeans  = MiniBatchKMeans(projection_dimensios,compute_labels=False)
        kmeans.fit(patches)
        components =  kmeans.cluster_centers_
    return components, mean_value


def image_to_components_space(image, components, mean_value, patches_size=(12,12)):
    patches = extract_patches_2d(image, patches_size)
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    sparse_coder = SparseCoder(components, transform_algorithm='threshold')
    encoded = []
    for s in gen_even_slices(len(patches), 100):
        data = patches[s] - mean_value
        encoded.extend(sparse_coder.transform(patches[s]))
    return numpy.asarray(encoded)


def plot_components(components):
    import cv2
    for i, comp in enumerate(components):
        c_img = (255 * minmaxnormalization(comp.reshape(12,12,3))).astype('uint8')
        imwrite('c%d.png'%i, c_img)


if __name__ == '__main__':
    from  cv2 import imread, imwrite
    import sys
    image = imread(sys.argv[1])
    sal = dictionary_saliency(image, 'ica', True)
    imwrite('saliency_ica.png', 255 * sal.astype('uint8'))
    del image
    del sal

    #image = imread(sys.argv[1])
    #sal = dictionary_saliency(image, 'kmeans')
    #imwrite('saliency_kmeans.jpg', 255 * sal.astype('uint8'))
