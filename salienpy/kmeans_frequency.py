import numpy
from sklearn.feature_extraction.image import extract_patches_2d
#from sklearn.decomposition import MiniBatchDictionaryLearning
#from sklearn.decomposition import FastICA
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import SparseCoder
from commons import gen_even_slices
from commons import minmaxnormalization



def kmeans_frequency_tuned_saliency(image):
    """
    Frequency Tuned Saliency.
    Args:
        image (numpy.array): an image.

    Returns:
       a 2d image saliency map.
    """
    components = extract_dictionary(image)
    encoded = image_to_dictionary_space(image, components)
    print "encoding patches shape = %s"% str(encoded.shape)
    encoded = encoded - encoded.mean(axis=0)
    encoded = encoded / abs(encoded).max()
    sal = numpy.sqrt(numpy.sum(encoded * encoded, axis=1))
    sal = numpy.reshape(sal, (image.shape[0] - 11, image.shape[1] - 11))
    return minmaxnormalization(sal)



def calculate_max_number_of_patches(image, patches_size=(12,12), max_number = 100000):
    fitting_patches = image.shape[0] * image.shape[1]
    fitting_patches = fitting_patches - (image.shape[0]*(patches_size[1] -1))
    fitting_patches = fitting_patches - (image.shape[1]*(patches_size[0] -1))
    return min(fitting_patches, max_number)


def extract_dictionary(image, patches_size=(12,12), projection_dimensios=12, previous_dictionary=None):
    """
    Gets a higher dimension ica projection image.

    """
    patches = extract_patches_2d(image, patches_size, calculate_max_number_of_patches(image, patches_size))
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    patches = (patches - patches.mean(axis=0))
    print "training patches shape = %s"% str(patches.shape)
    if not previous_dictionary:
        previous_dictionary = 'k-means++'
    kmeans  = MiniBatchKMeans(projection_dimensios,compute_labels=False,init=previous_dictionary)
    kmeans.fit(patches)
    return kmeans.cluster_centers_


def image_to_dictionary_space(image, dictionary, patches_size=(12,12)):
    patches = extract_patches_2d(image, patches_size)
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    patches = (patches - patches.mean(axis=0))
    sparse_coder = SparseCoder(dictionary, transform_algorithm='threshold')
    encoded = []
    for s in gen_even_slices(len(patches), 100):
        encoded.extend(sparse_coder.transform(patches[s]))
    return numpy.asarray(encoded)

if __name__ == '__main__':
    from  cv2 import imread, imwrite
    import sys
    image = imread(sys.argv[1])
    sal = kmeans_frequency_tuned_saliency(image)
    imwrite('saliency.jpg', sal)

    #import matplotlib
    #matplotlib.use('Agg')
    #from matplotlib import pyplot
    #components  = components -  components.min()
    #components  = 255* (components/components.max())

    #fig = pyplot.figure()
    #components = components.astype('uint8')
    #for i, c in enumerate(components):
    #    ax = fig.add_subplot(12, 2, i + 1)
    #    c = c.reshape((12,12,3))
    #    ax.imshow(c)
    #pyplot.savefig('out.png')
