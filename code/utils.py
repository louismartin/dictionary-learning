import numpy as np
import matplotlib.pyplot as plt

from nt_toolbox.signal import imageplot


def damage_image(image, removed_pixels):
    assert image.shape[0] == image.shape[1]
    img_size = image.shape[0]
    Omega = np.zeros([img_size, img_size])
    sel = np.random.permutation(img_size**2)
    np.ravel(Omega)[sel[np.arange(int(removed_pixels*img_size**2))]] = 1

    def Phi(f, Omega): return f*(1-Omega)

    damaged_image = Phi(image, Omega)
    return damaged_image


def random_dictionary(image, width, n_atoms):
    '''
    Takes an image as input and returns a dictionary
    of shape (width*width, n_atoms)
    '''
    assert image.shape[0] == image.shape[1]
    n0 = image.shape[0]

    # Random sampling of coordinates of the top left corner or the patches
    x = (np.random.random((1, 1, n_atoms))*(n0-width)).astype(int)
    y = (np.random.random((1, 1, n_atoms))*(n0-width)).astype(int)

    # Extract patches
    [dY, dX] = np.meshgrid(range(width), range(width))
    dX = np.tile(dX, (n_atoms, 1, 1)).transpose((1, 2, 0))
    dY = np.tile(dY, (n_atoms, 1, 1)).transpose((1, 2, 0))
    Xp = np.tile(x, (width, width, 1)) + dX
    Yp = dY + np.tile(y, (width, width, 1))
    D = image.flatten()[Yp+Xp*n0]
    D = D.reshape((width*width, n_atoms))  # Reshape from (w,w,q) to (w*w,q)
    return D


def center(D):
    '''
    Takes a Dictionary of shape (signal_size, n_atoms) and
    substract the signal-wise mean.
    '''
    assert len(D.shape) == 2
    D -= D.mean(axis=0)
    return D


def scale(D):
    ''' Scale the dictionary atoms to unit norm '''
    assert len(D.shape) == 2
    norm = np.tile(np.linalg.norm(D, axis=0), (D.shape[0], 1))
    D = np.divide(D, norm)
    return D


def high_energy_random_dictionary(image, width, n_atoms):
    '''
    Initialize a random dictionary with high energy centered and
    normalized atoms  of size (width*width, n_atoms)
    '''
    m = 20*n_atoms
    q = 3*m
    D = random_dictionary(image, width, q)
    D = center(D)
    # Keep patches with highest energy
    energies = np.sum(D**2, axis=0)
    Indexes = np.argsort(energies)[::-1]
    D = D[:, Indexes[:m]]
    # Select a random subset of these patches
    sel = np.random.permutation(range(m))[:n_atoms]
    D = D[:, sel]
    D = scale(D)
    return D


def reconstruction_error(Y, D, X):
    error = np.linalg.norm(Y - np.dot(D, X))**2
    return error


def plot_error(E, title='Reconstruction error', burn_in=None, filename=None):
    if burn_in:
        # Remove first points (burn in)
        E = E[2*burn_in:]

    E = np.log10(E)
    index = list(range(E.shape[0]))
    index_coef = list(range(0, E.shape[0], 2))
    index_dict = list(range(1, E.shape[0], 2))
    plt.plot(np.divide(index, 2), E)
    plt.plot(np.divide(index_coef, 2), E[index_coef],
             '*', markersize=7, label='After coefficient update')
    plt.plot(np.divide(index_dict, 2), E[index_dict],
             'o', markersize=5, label='After dictionary update')
    plt.legend(numpoints=1)
    plt.xlabel('Iterations')
    plt.ylabel('Error: $\log(||Y-DX||^2)$')
    plt.title(title)
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_dictionary(D, title='Dictionary'):
    ''' Plot a dictionary of shape (width*width, n_atoms) '''
    # Check that D.shape == (width*width, n_atoms)
    assert len(D.shape) == 2
    assert int(np.sqrt(D.shape[0]))**2 == D.shape[0]
    (signal_size, n_atoms) = D.shape
    width = int(np.sqrt(D.shape[0]))
    D = D.reshape((width, width, n_atoms))
    n = int(np.ceil(np.sqrt(n_atoms)))  # Size of the plot in number of atoms

    # Pad the atoms
    pad_size = 1
    missing_atoms = n ** 2 - n_atoms

    padding = (((pad_size, pad_size), (pad_size, pad_size),
                (0, missing_atoms)))
    D = np.pad(D, padding, mode='constant', constant_values=1)
    padded_width = width + 2*pad_size
    D = D.reshape(padded_width, padded_width, n, n)
    D = D.transpose(2, 0, 3, 1)  # Needed for the reshape
    big_image_size = n*padded_width
    D = D.reshape(big_image_size, big_image_size)
    plt.figure(figsize=(8, 12))
    imageplot(D)
    plt.title(title)
    plt.show()
