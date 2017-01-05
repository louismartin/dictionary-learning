
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


# # Dictionary learning: numerical tour approach
# TODO: quote the matlab numerical tour

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
    ''' Scale the dictionary atoms to unit norm '''
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

def plot_dictionary(D):
    ''' Plot a dictionary of shape (width*width, n_atoms) '''
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

def ProjX(X,k):
    ''' Sparsity projection, keeps the k largest coefficients '''
    X = X * (abs(X) >= np.sort(abs(X), axis=0)[-k,:])
    return X


def ProjC(D):
    ''' Dictionary projection, scales the atoms '''
    D = scale(D)
    return D

def sparse_code_pgd(Y, D, X, sparsity=4, n_iter=100):
    '''
    Sparse code data Y using dictionary D using a forward backward iterative scheme.
    This is a non-smooth and non-convex minimization, that can be shown to be NP-hard.
    A heuristic to solve this method is to compute a stationary point of the energy
    using the Foward-Backward iterative scheme (projected gradient descent).
    '''
    gamma = 1/np.linalg.norm(np.dot(D,D.T)) # TODO: Improve gamma ? (compare with nt)
    for i in range(n_iter):
        R = np.dot(D, X) - Y
        X = ProjX(X - gamma * np.dot(D.T, R), sparsity)
    return X

def sparse_code_lasso(Y, D, model):
    ''' Sparse code data Y using dictionary D using lasso linear regression '''
    X = lasso.fit(D, Y).coef_.T
    return X

def dictionary_update_pgd(Y, D, X, n_iter=50):
    tau = 1/np.linalg.norm(np.dot(X, X.T)) # TODO: Improve tau ? (compare with nt)
    for i in range(n_iter):
        R = np.dot(D, X) - Y
        D = ProjC(D - tau * np.dot(R, X.T))
    return D


img_size = 256
filename = 'image.jpg'
filename = 'barb_crop.png'
filename = 'lena.bmp'
f0 = load_image(filename, img_size)

plt.figure(figsize = (6,6))
imageplot(f0, 'Image f_0')


width = 10
signal_size = width*width
n_atoms = 2*signal_size
n_test = 20*n_atoms
k = 4


D0 = high_energy_random_dictionary(f0, width, n_atoms)
Y_test = random_dictionary(f0, width, n_test)


n_iter_learning = 6
n_iter_dico = 25
n_iter_coef = 50
E = np.zeros(2*n_iter_learning)
X = np.zeros((n_atoms, n_test))
D = D0
for i in tqdm(range(n_iter_learning)):
    # --- coefficient update ----
    X = sparse_code_pgd(Y_test, D, X, sparsity=k, n_iter=n_iter_coef)
    E[2*i] = np.linalg.norm(Y_test - np.dot(D, X))**2
    # --- dictionary update ----
    D = dictionary_update_pgd(Y_test, D, X, n_iter=n_iter_dico)
    E[2*i+1] = np.linalg.norm(Y_test - np.dot(D, X))**2


# Remove first points (burn in)
start = 4
assert start%2==0
E_plot = E[start:]

plt.plot(range(E_plot.shape[0]), E_plot)
index_coef = list(range(0, E_plot.shape[0], 2))
index_dico = list(range(1, E_plot.shape[0], 2))
plt.plot(index_coef, E_plot[index_coef], '*', label='After coefficient update')
plt.plot(index_dico, E_plot[index_dico], 'o', label='After dictionary update')
plt.legend(numpoints=1)
plt.title('$J(x)$')
plt.show()

plt.figure(figsize=(8,12))
plot_dictionary(D0)
plt.title('D0')
plt.show()
plt.figure(figsize=(8,12))
plot_dictionary(D)
plt.title('D')
plt.show()

min(E)


n_test = 1000
Y_test = random_dictionary(f0, width, n_test)


X0 = np.zeros((n_atoms, n_test))
tic = time.time()
for j in range(n_iter_coef):
    R = np.dot(D, X0) - Y_test
    X0 = ProjX(X0 - gamma * np.dot(D.T, R), k)
print(time.time()-tic)
print(np.sum(X0!=0, axis=0))
np.linalg.norm(Y_test - np.dot(D, X0))**2


np.linalg.norm(Y_test - np.dot(D, X0))**2


lasso = linear_model.Lasso(0.01, fit_intercept=False)
tic = time.time()
X1 = lasso.fit(D, Y_test).coef_.T
print(time.time()-tic)
print(np.sum(X1!=0, axis=0))
np.linalg.norm(Y_test - np.dot(D, X1))**2


np.mean(np.sum(X1!=0, axis=0))


X1[:,0].shape


# We construct a mask $\Omega$ made of random pixel locations.  
# The damaging operator put to zeros the pixel locations $x$ for which $\Omega(x)=1$.  
# The damaged observations reads $y = \Phi f_0$.

def damage_image(image):
    rho = .7 # percentage of removed pixels
    Omega = np.zeros([img_size, img_size])
    sel = np.random.permutation(img_size**2)
    np.ravel(Omega)[sel[np.arange(int(rho*img_size**2))]] = 1

    Phi = lambda f, Omega: f*(1-Omega)

    damaged_image = Phi(image, Omega)
    return damaged_image

y = damage_image(f0)
plt.figure(figsize = (6,6))
imageplot(y, 'Observations y')


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
    X = model.fit(D, Y_test).coef_
    error = np.linalg.norm(Y_test - np.dot(D,X.T))
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

D0 = high_energy_random_dictionary(f0, width, n_atoms) # Initialize dictionary with k random atoms
D = D0
# TODO: normalize atom to unit norm as sparsity_4_dictionary_learning ?
A = np.zeros((n_atoms,n_atoms))
B = np.zeros((signal_size,n_atoms))

# Evaluation data initialization
sparsity = []
error = []
Y_test = random_dictionary(f0, width, 20*n_atoms)

for t in tqdm(range(T)):
    y = random_dictionary(f0, width, n_atoms=1) # Draw 1 random patch as column vector
    x = lasso.fit(D, y).coef_.reshape((n_atoms,1)) # Get the sparse coding # TODO: try with lasso.sparse_coef_
    A += np.dot(x,x.T)
    B += np.dot(y,x.T)
    D = update_dictionary(D, A, B)
    
    if t%10 == 0:
        # Evaluation:
        error.append(evaluate(Y_test, D, lasso))
        sparsity.append(np.sum(x!=0))#/alpha.shape[0]


plt.plot(error)
plt.show()
plt.figure(figsize=(8,12))
plot_dictionary(D0)
plt.title('D0')
plt.show()
plt.figure(figsize=(8,12))
plot_dictionary(D)
plt.title('D')
plt.show()





X = np.array([[1,2],[2,3],[-1,3]]).T
1/np.linalg.norm(np.dot(X, X.T))


np.sqrt(np.sum(np.dot(X, X.T)**2))


X = np.array([[1,2,3], [4,3,2], [-1,5,3],[-3,-8,2]]).T

