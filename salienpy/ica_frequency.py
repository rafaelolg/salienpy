import numpy
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning

def extract_ica(image, patches_size=(10,10), projection_dimensios):
    """
    Gets a higher dimension ica projection image.

    """
    patches = image.extract_patches_2d(image, patches_size)
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    patches -= numpy.mean(patches, axis=0)
    patches /= numpy.std(patches, axis=0)
    dico = MiniBatchDictionaryLearning(n_atoms=projection_dimensios, alpha=1, n_iter=500)
    fit = dico.fit(r)
    ####TODO: maybe this is not needed.
    components = V.components_.reshape(projection_dimensios, patches_size[0],
                                                             patches_size[1],
                                                             image.shape[2])
    