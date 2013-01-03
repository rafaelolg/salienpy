import numpy
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from matplotlib import pyplot

def extract_dictionary(image, patches_size=(12,12), projection_dimensios=24, previous_dictionary=None):
    """
    Gets a higher dimension ica projection image.

    """
    patches = extract_patches_2d(image, patches_size, max_patches=0.7)
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    patches = (patches - patches.mean(axis=0))# / numpy.std(patches, axis=0)
    print "training patches shape = %s"% str(patches.shape)
    dico = MiniBatchDictionaryLearning(n_atoms=projection_dimensios, alpha=1, n_iter=100, transform_algorithm='threshold')
    fit = dico.fit(patches)
    return fit

def image_to_dictionary_space(image, dictionary, patches_size=(12,12)):
    patches = extract_patches_2d(image, patches_size)
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    patches = (patches - patches.mean(axis=0))# / numpy.std(patches, axis=0)
    dict_encoded = dictionary.transform(patches)
    print "encoding patches shape = %s"% str(patches.shape)
    return dict_encoded

if __name__ == '__main__':
    from  cv2 import imread, imwrite
    import sys
    image = imread(sys.argv[1])
    d = extract_dictionary(image)
    encoded = image_to_dictionary_space(image, d)

    components  = d.components_ -  d.components_.min()
    components  = 255* (components/components.max())
    components = components.astype('uint8')
    for i, c in enumerate(components):
        pyplot.subplot(12, 12, i + 1)
        c = c.reshape((12,12,3)) 
        pyplot.imshow(c)