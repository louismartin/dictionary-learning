
# coding: utf-8

# # Online Matrix Factorization

# Code taken from  
# http://nbviewer.jupyter.org/github/gpeyre/numerical-tours/blob/master/python/inverse_5_inpainting_sparsity.ipynb

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

from nt_toolbox.signal import load_image, imageplot, snr
from nt_toolbox.general import clamp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


# Here we consider inpainting of damaged observation without noise.

img_size = 256
#f0 = load_image('image.jpg', img_size)
f0 = load_image('lena.bmp', img_size)

plt.figure(figsize = (6,6))
imageplot(f0, 'Image f_0')


# We construct a mask $\Omega$ made of random pixel locations.  
# The damaging operator put to zeros the pixel locations $x$ for which $\Omega(x)=1$.  
# The damaged observations reads $y = \Phi f_0$.

from numpy import random

rho = .7 # percentage of removed pixels
Omega = np.zeros([img_size, img_size])
sel = random.permutation(img_size**2)
np.ravel(Omega)[sel[np.arange(int(rho*img_size**2))]] = 1

Phi = lambda f, Omega: f*(1-Omega)

y = Phi(f0, Omega)

plt.figure(figsize = (6,6))
imageplot(y, 'Observations y')


# ### Algorithm 1

# Dictionary initialization inspired from  
# http://nbviewer.jupyter.org/github/gpeyre/numerical-tours/blob/master/matlab/sparsity_4_dictionary_learning.ipynb

w = 10   # Width of the patches
m = w*w  # Size of the signal to be sparse coded
k = 2*m  # Number of atoms in the dictionary (overcomplete)


# Generate a random patch in the damaged image

def random_patch(image, width, n_patches=1):
    img_shape = image.shape
    # Upper left corners of patches
    rows = np.random.randint(0, img_shape[0]-width, n_patches)
    cols = np.random.randint(0, img_shape[1]-width, n_patches)
    
    patches = np.zeros((width*width, n_patches))
    for i in range(n_patches):
        patch = image[
            rows[i]:rows[i]+width,
            cols[i]:cols[i]+width
        ]
        patches[:,i] = patch.flatten()
    
    return patches

def plot_dictionary(D):
    '''
    Plot a dictionary of shape (width*width, n_atoms)
    '''
    # Check that D.shape == (width*width, n_atoms)
    assert len(D.shape) == 2
    assert int(np.sqrt(D.shape[0]))**2 == D.shape[0]
    (signal_size, n_atoms) = D.shape
    width = int(np.sqrt(D.shape[0]))
    D = D.reshape((width,width,n_atoms))
    n = int(np.ceil(np.sqrt(n_atoms))) # Size of the plot square in number of atoms

    # Pad the atoms
    pad_size = 1
    missing_atoms = n ** 2 - n_atoms

    padding = (((pad_size, pad_size), (pad_size, pad_size),
                (0, missing_atoms)))
    D = np.pad(D, padding, mode='constant', constant_values=1)
    padded_width = width + 2*pad_size
    D = D.reshape(padded_width,padded_width,n,n)
    D = D.transpose(2,0,3,1) # Needed for the reshape
    big_image_size = n*padded_width
    D = D.reshape(big_image_size, big_image_size)
    imageplot(D)


# ### Algorithm 2 Dictionary update
# From "Online Learning for Matrix Factorization and Sparse Coding"

def update_dictionary(D, A, B):
    '''
    Update the dictionary column by column.
    Denoting k the number of atoms in the dictionary and m the size of the signal, we have:
    
    Args:
        D: dictionary of size (m,k)
        A: Matrix of size (k,k)
        B: Matrix of size (m,k)
    Returns:
        D: Updated dictionary of size (m,k)
    '''
    (m,k) = D.shape
    assert A.shape == (k,k)
    assert B.shape == (m,k)
    
    for j in range(k):        
        uj = (B[:,j]-np.dot(D,A[:,j])) + D[:,j]
        if A[j,j] != 0:
            uj /= A[j,j]
        else:
            # TODO: What to do when A[j,j] is 0 ?
            pass
        D[:,j] = 1/max(np.linalg.norm(uj),1)*uj
    return D


def evaluate(Y_test, D, model):
    alpha = model.fit(D, Y_test).coef_
    error = np.linalg.norm(Y_test - np.dot(D,alpha.T))
    #score = model.score(D, Y_test)
    return error


# ### Algorithm 1 Online dictionary learning
# From "Online Learning for Matrix Factorization and Sparse Coding"

import time
from tqdm import tqdm
from sklearn import linear_model

# Initialize variables
T = 100 # Number of iterations
lambd = 0.1 # L1 penalty coefficient for alpha
# LARS-Lasso from LEAST ANGLE REGRESSION, Efron et al http://statweb.stanford.edu/~tibs/ftp/lars.pdf
lasso = linear_model.Lasso(lambd, fit_intercept=False) # TODO: use lars instead of lasso

D = high_energy_random_dictionary(f0, w, n_atoms=k) # Initialize dictionary with k random atoms
# TODO: normalize atom to unit norm as sparsity_4_dictionary_learning ?
A = np.zeros((k,k))
B = np.zeros((m,k))

# Evaluation data initialization
sparsity = []
error = []
n = 20*k # Number of patch to take for evaluation
Y_test = random_dictionary(f0, w, n_atoms=n)

plt.figure(figsize=(8,12))
plot_dictionary(D)

start = time.time()
for t in tqdm(range(T)):
    x = random_dictionary(f0, w, n_atoms=1) # Draw 1 random patch as column vector
    alpha = lasso.fit(D, x).coef_.reshape((k,1)) # Get the sparse coding # TODO: try with lasso.sparse_coef_
    A += np.dot(alpha,alpha.T)
    B += np.dot(x,alpha.T)
    D = update_dictionary(D, A, B)
    
    if t%10 == 0:
        # Evaluation:
        error.append(evaluate(Y_test, D, lasso))
        sparsity.append(np.sum(alpha!=0))#/alpha.shape[0]
end = time.time()

print('Time elapsed: %.3f s' % (end-start))


plt.figure(figsize=(8,12))
plot_dictionary(D)
plt.figure(figsize=(8,12))
plt.plot(error)
plt.show()








def random_dictionary(image, width, n_atoms):
    '''
    Takes an image as input and returns a dictionary
    of shape (width*width, n_atoms)
    '''
    assert image.shape[0] == image.shape[1]
    n0 = image.shape[0]
    
    # Random sampling of coordinates of the top left corner or the patches
    x = (np.random.random((1,1,n_atoms))*(n0-width)).astype(int)
    y = (np.random.random((1,1,n_atoms))*(n0-width)).astype(int)
    
    # Extract patches
    [dY,dX] = np.meshgrid(range(width), range(width))
    dX = np.tile(dX, (n_atoms,1,1)).transpose((1,2,0))
    dY = np.tile(dY, (n_atoms,1,1)).transpose((1,2,0))
    Xp = np.tile(x, (width,width,1)) + dX
    Yp = dY + np.tile(y, (width,width,1))
    D = image.flatten()[Yp+Xp*n0]
    D = D.reshape((width*width,n_atoms)) # Reshape from (w,w,q) to (w*w,q)
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
    '''
    Scale the dictionary atoms to unit norm
    '''
    assert len(D.shape) == 2
    norm = np.tile(np.linalg.norm(D, axis=0), (D.shape[0],1))
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
    D = D[:,Indexes[:m]]
    # Select a random subset of these patches
    sel = np.random.permutation(range(m))[:n_atoms]
    D = D[:,sel]
    D = scale(D)
    return D


img_size = 256
f0 = load_image('barb_crop.png', img_size)

#plt.figure(figsize = (6,6))
imageplot(f0, 'Image f_0')
n0 = f0.shape[0]


width = 10
signal_size = width*width
n_atoms = 2*signal_size
n_test = 20*n_atoms
k = 4


D0 = high_energy_random_dictionary(f0, width, n_atoms)
Y = random_dictionary(f0, width, n_test)


# Sparsity projection, keeps the k largest coefficients
ProjX = lambda X,k: X * (abs(X) >= np.sort(abs(X))[-k])


projC = scale

n_iter_learning = 1
n_iter_dico = 50
n_iter_coef = 100
E = np.zeros(2*n_iter_learning)
X = np.zeros((n_atoms, n_test))
D = D0
pbar = tqdm(total=n_iter_learning*(n_iter_dico+n_iter_coef))
for i in range(n_iter_learning):
    # --- coefficient update ----
    gamma = 1.6/np.linalg.norm(D)**2
    for j in range(n_iter_coef):
        R = np.dot(D, X) - Y
        X = ProjX(X - gamma * np.dot(D.T, R), k)
        pbar.update(1)
    E[2*i] = np.linalg.norm(Y - np.dot(D, X))**2
    # --- dictionary update ----
    tau = 1/np.linalg.norm(np.dot(X, X.T)) # TODO: check if it is the same value
    for j in range(n_iter_dico):
        R = np.dot(D, X) - Y
        D = ProjC(D - tau * np.dot(R, X.T))
        pbar.update(1)
    E[2*i+1] = np.linalg.norm(Y - np.dot(D, X))**2
pbar.close()


plt.plot(range(2*n_iter_learning), E)
index_coef = list(range(0, 2*n_iter_learning, 2))
index_dico = list(range(1, 2*n_iter_learning, 2))
plt.plot(index_coef, E[index_coef], '*', label='After coefficient update')
plt.plot(index_dico, E[index_dico], 'o', label='After dictionary update')
plt.legend(numpoints=1)
plt.title('$J(x)$')
plt.show()




