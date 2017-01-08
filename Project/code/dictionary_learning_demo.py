
# coding: utf-8

# # Dictionary learning

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

import time
import shelve
import pickle
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn import linear_model, datasets

from nt_toolbox.signal import load_image, imageplot, snr
from nt_toolbox.general import clamp
from utils import (random_dictionary, high_energy_random_dictionary,
                  center, scale, reconstruction_error, plot_error, plot_dictionary)
from dictionary_learning import (dictionary_update_ksvd, sparse_code_lasso, dictionary_update_omf,
                                 sparse_code_fb, dictionary_update_fb)

import warnings
warnings.filterwarnings('ignore')


# ## Image and variables

img_size = 256
filename = 'lena.bmp'
f0 = load_image(filename, img_size)

#plt.figure(figsize = (6,6))
#imageplot(f0, 'Image f_0')


width = 5
signal_size = width*width
n_atoms = 2*signal_size
n_samples = 20*n_atoms
k = 4 # Desired sparsity


synthetic_data = True
if synthetic_data:
    Y, D0, X0 = datasets.make_sparse_coded_signal(n_samples, n_atoms, signal_size, k, random_state=0)
else:
    D0 = high_energy_random_dictionary(f0, width, n_atoms)
    Y = random_dictionary(f0, width, n_samples)
    Y = center(Y) # TODO: Center because the dictionary is centered and no intercept


# # K-SVD
# 
# Aharon, Michal, Michael Elad, and Alfred Bruckstein. "K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation." IEEE Transactions on signal processing 54.11 (2006): 4311-4322.

n_iter = 10
E = np.zeros(2*n_iter)
X = np.zeros((n_atoms, n_samples))
D = np.random.random(D0.shape)

for i in tqdm(range(n_iter)):
    # Sparse coding
    X = sparse_code_fb(Y, D, X, sparsity=k, n_iter=100)
    E[2*i] = reconstruction_error(Y, D, X)

    # Dictionary update
    D, _ = dictionary_update_ksvd(Y, D, X)
    #D = dictionary_update_fb(Y, D, X, n_iter=50)
    E[2*i+1] = reconstruction_error(Y, D, X)


plot_error(E)


# # Forward Backward
# 
# Combettes, Patrick L., and Jean-Christophe Pesquet. "Proximal splitting methods in signal processing." Fixed-point algorithms for inverse problems in science and engineering. Springer New York, 2011. 185-212.  
# 
# Adapted from
# http://nbviewer.jupyter.org/github/gpeyre/numerical-tours/blob/master/matlab/sparsity_4_dictionary_learning.ipynb

n_iter_learning = 5
n_iter_dico = 50
n_iter_coef = 100
E = np.zeros(2*n_iter_learning)
X = np.zeros((n_atoms, n_samples))
D = D0
for i in tqdm(range(n_iter_learning)):
    # Sparse coding
    X = sparse_code_fb(Y, D, X, sparsity=k, n_iter=n_iter_coef)
    E[2*i] = reconstruction_error(Y, D, X)

    # Dictionary update
    D = dictionary_update_fb(Y, D, X, n_iter=n_iter_dico)
    E[2*i+1] = reconstruction_error(Y, D, X)


plot_error(E)
plot_dictionary(D0)


# # Online dictionary learning
# From "Online Learning for Matrix Factorization and Sparse Coding"  
# LARS-Lasso from LEAST ANGLE REGRESSION, Efron et al http://statweb.stanford.edu/~tibs/ftp/lars.pdf

n_iter = 5*n_samples
test_interval = 1000
lambd = 0.01 # L1 penalty coefficient for sparse coding
lasso = linear_model.Lasso(lambd, fit_intercept=False) # TODO: use lars instead of lasso

D = D0
A = np.zeros((n_atoms,n_atoms))
B = np.zeros((signal_size,n_atoms))

sparsity = []
E = []
Y = random_dictionary(f0, width, 20*n_atoms)
X = np.zeros((n_atoms, n_samples))


for i in tqdm(range(n_iter)):
    # Draw 1 random patch y and get its sparse coding
    #y = random_dictionary(f0, width, n_atoms=1)
    y = Y[:,np.random.randint(Y.shape[1])].reshape((signal_size,1))
    x = lasso.fit(D, y).coef_.reshape((n_atoms,1))
    A += np.dot(x,x.T)
    B += np.dot(y,x.T)
    D = dictionary_update_omf(D, A, B)
    D = scale(D)
    sparsity.append(np.mean(np.sum(x!=0, axis=0)))
    
    if i%test_interval == 0:
        # Evaluation:
        X = sparse_code_fb(Y, D, X, sparsity=4, n_iter=100)
        E.append(reconstruction_error(Y, D, X))
        #sparsity.append(np.mean(np.sum(X!=0, axis=0)))


plt.plot(range(0, n_iter, test_interval), E)
plt.title('Reconstruction error on the test set')
plt.show()
#plt.savefig('omf_2400000_iter.png')
plt.figure(figsize=(8,12))
plot_dictionary(D0)
plt.title('D0')
plt.show()
plt.figure(figsize=(8,12))
plot_dictionary(D)
plt.title('D')
plt.show()
plt.plot(sparsity)


# Save variables

filename = op.join('vars','omf_iter.out')
with shelve.open(filename,'n') as shelf: # 'n' for new
    for key in dir():
        try:
            shelf[key] = globals()[key]
        except (TypeError, pickle.PicklingError, AttributeError):
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))











import numpy as np
x1 = np.array([[1,2,3]]).T
x2 = np.array([[2,2,2]]).T
X = np.array(np.hstack((x1,x2)))
print(X.T)
print(X)
print(X.shape)
print(np.dot(X, X.T))
print(np.dot(x1,x1.T)+np.dot(x2,x2.T))


print(np.dot(x1,x1.T))




