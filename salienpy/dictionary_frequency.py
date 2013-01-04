import numpy
from sklearn.feature_extraction.image import extract_patches_2d
#from sklearn.decomposition import MiniBatchDictionaryLearning
#from sklearn.decomposition import FastICA
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot

def extract_dictionary(image, patches_size=(12,12), projection_dimensios=12, previous_dictionary=None):
    """
    Gets a higher dimension ica projection image.

    """
    patches = extract_patches_2d(image, patches_size, max_patches=200000)
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    #patches = (patches - patches.mean(axis=0))# / numpy.std(patches, axis=0)
    print "training patches shape = %s"% str(patches.shape)
    if not previous_dictionary:
        previous_dictionary = 'k-means++'
    kmeans  = MiniBatchKMeans(projection_dimensios,compute_labels=False,init=previous_dictionary)
    kmeans.fit(patches)
    return kmeans.cluster_centers_

def image_to_dictionary_space(image, dictionary, patches_size=(12,12)):
    patches = extract_patches_2d(image, patches_size)
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    #patches = (patches - patches.mean(axis=0))# / numpy.std(patches, axis=0)
    dict_encoded = dictionary.transform(patches)
    print "encoding patches shape = %s"% str(patches.shape)
    return dict_encoded

if __name__ == '__main__':
    from  cv2 import imread, imwrite
    import sys
    image = imread(sys.argv[1])
    components = extract_dictionary(image)
    #encoded = image_to_dictionary_space(image, d)
    #components  = d -  d.min()
    #components  = 255* (components/components.max())
    fig = pyplot.figure()
    components = components.astype('uint8')
    import matplotlib
    matplotlib.use('Agg')
    for i, c in enumerate(components):
        ax = fig.add_subplot(12, 2, i + 1)
        c = c.reshape((12,12,3))
        ax.imshow(c)
    pyplot.savefig('out.png')
