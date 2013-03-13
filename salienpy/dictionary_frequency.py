import numpy
from sklearn.feature_extraction.image import extract_patches_2d

from sklearn.decomposition import FastICA
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import SparseCoder
from commons import gen_even_slices
from commons import minmaxnormalization
from multiscale import multiscale_saliency


DEFAULT_PATCHE_SIZE = 16

def dictionary_saliency(image,algorithm='ica', show_info=False):
    """
    Dictionary frequency Tuned Saliency.
    Args:
        image (numpy.array): an image.
        algorithm: the algorithm to create the sparse dictionary .
            Accepted values  are kmeans and ica.
    Returns:
       a 2d image saliency map in z-value space.
    """
    method = lambda img: _dictionary_saliency(img, algorithm, show_info)
    return _dictionary_saliency(image, algorithm, show_info)


def _dictionary_saliency(image,algorithm='ica', show_info=False):
    from time import time
    t0 = time()
    encoder = calculate_encoder(image, algorithm)
    train_time = time()
    encoded = image_to_components_space(image, encoder)
    ecoding_time = time()
    #distance to mean value
    encoded = encoded - encoded.mean(axis=0)
    sal = numpy.sqrt(numpy.sum(encoded * encoded, axis=1))
    sal = numpy.reshape(sal, (image.shape[0] - (DEFAULT_PATCHE_SIZE-1),
                              image.shape[1] -(DEFAULT_PATCHE_SIZE-1)))
    #saliency in z-value value
    sal = (sal - sal.mean())/sal.std()
    frequency_diff_time = time()
    total_time = time()
    if show_info:
        print '(train_time:%s, ecoding_time:%s, frequency_diff_time:%s, total_time: %s)' % (
                train_time - t0,
                ecoding_time - train_time,
                frequency_diff_time - ecoding_time,
                total_time-t0)
        plot_components(extract_components(encoder, algorithm),(DEFAULT_PATCHE_SIZE,DEFAULT_PATCHE_SIZE))
    return sal


def extract_components(encoder, algorithm):
    if algorithm == 'ica':
        return encoder.components_


def calculate_max_number_of_patches(image, patches_size=(DEFAULT_PATCHE_SIZE,DEFAULT_PATCHE_SIZE), max_number = 10000):
    fitting_patches = image.shape[0] * image.shape[1]
    fitting_patches = fitting_patches - (image.shape[0]*(patches_size[1] -1))
    fitting_patches = fitting_patches - (image.shape[1]*(patches_size[0] -1))
    return min(fitting_patches, max_number)


def calculate_encoder(image,
                       algorithm,
                       patches_size=(DEFAULT_PATCHE_SIZE,DEFAULT_PATCHE_SIZE),
                       projection_dimensios=DEFAULT_PATCHE_SIZE,
                       previous_components=None):
    """
    Gets a higher dimension sparse dictionary.algorithm can be
    ica, kmeans or colors.

    """
    patches = extract_patches_2d(image, patches_size,
            calculate_max_number_of_patches(image, patches_size) )
    patches = numpy.reshape(patches, (patches.shape[0],-1)).astype(float)
    encoder = None
    if algorithm == 'ica':
        encoder = FastICA(n_components=projection_dimensios, whiten=True, max_iter=10)
    elif algorithm == 'kmeans':
        encoder  = MiniBatchKMeans(projection_dimensios,compute_labels=False)
    else:
        raise Exception('Unknow algorithm %s' % algorithm)
    encoder.fit(patches)
    return encoder


def image_to_components_space(image, encoder,  patches_size=(DEFAULT_PATCHE_SIZE,DEFAULT_PATCHE_SIZE)):
    patches = extract_patches_2d(image, patches_size)
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    encoded = []
    for s in gen_even_slices(len(patches), 100):
        encoded.extend(encoder.transform(patches[s]))
    return numpy.asarray(encoded)


def plot_components(components, patches_size=(DEFAULT_PATCHE_SIZE,DEFAULT_PATCHE_SIZE)):
    import cv2
    for i, comp in enumerate(components):
        c_img = (255 * minmaxnormalization(comp.reshape(patches_size[0],patches_size[1],3))).astype('uint8')
        imwrite('c%d.png'%i, c_img)


if __name__ == '__main__':
    from  cv2 import imread, imwrite
    import sys
    image = imread(sys.argv[1])
    sal = dictionary_saliency(image, 'ica', True)
    sal = 255 * minmaxnormalization(sal)
    imwrite('saliency_ica.png', sal.astype('uint8'))
    del image
    del sal

    #image = imread(sys.argv[1])
    #sal = dictionary_saliency(image, 'kmeans')
    #imwrite('saliency_kmeans.jpg', (255 * sal).astype('uint8'))

