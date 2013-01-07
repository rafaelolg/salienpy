import numpy
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import FastICA
from sklearn.decomposition import SparseCoder
from commons import gen_even_slices
from commons import minmaxnormalization




def aws_saliency(image, show_info=False):
    """
    Frequency Tuned Saliency.
    Args:
        image (numpy.array): an image.

    Returns:
       a 2d image saliency map.
    """
    from time import time
    t0 = time()
    components, mean_value = extract_components(image)
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


def extract_components(image, patches_size=(12,12), projection_dimensios=12, previous_components=None):
    """
    Gets a higher dimension ica projection image.

    """
    patches = extract_patches_2d(image, patches_size, calculate_max_number_of_patches(image, patches_size))
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    mean_value  = patches.mean(axis=0)
    patches = (patches - mean_value)
    ica = FastICA(n_components=projection_dimensios, whiten=True, max_iter=10)
    ica.fit(patches.T)
    return ica.components_.T, mean_value


def image_to_components_space(image, components, mean_value, patches_size=(12,12)):
    patches = extract_patches_2d(image, patches_size)
    patches = numpy.reshape(patches, (patches.shape[0],-1))
    patches = (patches - mean_value)
    sparse_coder = SparseCoder(components, transform_algorithm='threshold')
    encoded = []
    for s in gen_even_slices(len(patches), 100):
        encoded.extend(sparse_coder.transform(patches[s]))
    return numpy.asarray(encoded)


def plot_components(components):
    n_col = 3
    n_row = 4
    from matplotlib import pyplot as pl
    pl.figure(figsize=(2. * n_col, 2.26 * n_row))
    pl.suptitle('Components', size=16)
    for i, comp in enumerate(components):
        pl.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        pl.imshow(comp.reshape(12,12,3),
                  interpolation='nearest',
                  vmin=-vmax, vmax=vmax)
        pl.xticks(())
        pl.yticks(())
    import ipdb;ipdb.set_trace()
    pl.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    pl.savefig('components.png')


if __name__ == '__main__':
    from  cv2 import imread, imwrite
    import sys
    image = imread(sys.argv[1])
    sal = aws_saliency(image)
    import ipdb;ipdb.set_trace()
    imwrite('saliency.jpg', 255 * sal.astype('uint8'))
