
# coding: utf-8

# # Dictionary learning

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

import time

import numpy as np
from tqdm import tqdm
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from nt_toolbox.signal import load_image, imageplot
from utils import (random_dictionary, high_energy_random_dictionary,
                  center, scale, reconstruction_error, plot_error, plot_dictionary)
from dictionary_learning import (sparse_code_omp, dictionary_update_ksvd, sparse_code_lasso,
                                 dictionary_update_omf, sparse_code_fb, dictionary_update_fb)

import warnings
warnings.filterwarnings('ignore')


# ## Image and variables

width = 5
signal_size = width*width
n_atoms = 2*signal_size
n_samples = 20*n_atoms
k = 4 # Desired sparsity
max_time = 120


synthetic_data = True
if synthetic_data:
    Y, D_true, X_true = datasets.make_sparse_coded_signal(n_samples, n_atoms, signal_size, k, random_state=0)
else:
    img_size = 256
    filename = 'images/lena.bmp'
    f0 = load_image(filename, img_size)

    plt.figure(figsize = (6,6))
    imageplot(f0, 'Image f_0')

    D0 = high_energy_random_dictionary(f0, width, n_atoms)
    Y = random_dictionary(f0, width, n_samples)
    Y = center(Y) # TODO: Center because the dictionary is centered and no intercept

np.random.seed(0)
D0 = np.random.random(D_true.shape)
X0 = np.zeros((n_atoms, n_samples))


# ### Performance considerations

# Sparse coding methods

print('Orthogonal matching pursuit')
omp = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
get_ipython().magic(u'timeit sparse_code_omp(Y, D_true, omp)')
X_omp = sparse_code_omp(Y, D_true, omp)
print('Reconstruction error: {}'.format(reconstruction_error(Y, D_true, X_omp)))

print('\nLasso')
lasso = linear_model.Lasso(0.01, fit_intercept=False)
get_ipython().magic(u'timeit sparse_code_lasso(Y, D_true, lasso)')
X_lasso = sparse_code_lasso(Y, D_true, lasso)
print('Mean sparsity: {}'.format(np.mean(np.sum(X_lasso != 0, axis=0))))
print('Reconstruction error: {}'.format(reconstruction_error(Y, D_true, X_lasso)))

print('\nForward backward')
get_ipython().magic(u'timeit sparse_code_fb(Y, D_true, X0, sparsity=k, n_iter=100)')
X_fb = sparse_code_fb(Y, D_true, X0, sparsity=k, n_iter=100)
print('Reconstruction error: {}'.format(reconstruction_error(Y, D_true, X_fb)))


# Dictionary learning methods
# 
# Results are not comparable, this only gives a rough idea of the time algorithm take

print('K-SVD')
get_ipython().magic(u'timeit dictionary_update_ksvd(Y, D, X_true)')

print('\nOnline matrix factorization')
A = np.zeros((n_atoms,n_atoms))
B = np.zeros((signal_size,n_atoms))
y = Y[:, 0].reshape((signal_size, 1))
get_ipython().magic(u'timeit dictionary_update_omf(D, A, B)')

print('\nForward backward')
get_ipython().magic(u'timeit dictionary_update_fb(Y, D, X_true, n_iter=50)')


# # K-SVD
# 
# Aharon, Michal, Michael Elad, and Alfred Bruckstein. "K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation." IEEE Transactions on signal processing 54.11 (2006): 4311-4322.

# Variables
n_iter = 100
D = D0.copy()
X = X0.copy()
E = np.zeros(2*n_iter)
times = np.zeros(2*n_iter)

# Model for sparse coding
omp = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
tic = time.time()
for i in tqdm(range(n_iter)):
    # Sparse coding
    X = sparse_code_omp(Y, D, omp)
    E[2*i] = reconstruction_error(Y, D, X)
    times[2*i] = time.time() - tic

    # Dictionary update
    D, X = dictionary_update_ksvd(Y, D, X)
    E[2*i+1] = reconstruction_error(Y, D, X)
    times[2*i+1] = time.time() - tic
    if (time.time() - tic) > max_time:
        break

# Save variables
E_ksvd = E.copy()
times_ksvd = times.copy()
D_ksvd = D.copy()


plot_error(E, burn_in=2, filename='images/ksvd_{}_iter_{}.png'.format(n_iter, 'synthetic' if synthetic_data else 'image'))
#plot_dictionary(D)


# # Online dictionary learning
# Mairal, Julien, et al. "Online learning for matrix factorization and sparse coding." Journal of Machine Learning Research 11.Jan (2010): 19-60.

# Variables
n_iter = 10*n_samples
eval_interval = n_samples//2
D = D0.copy()
X = X0.copy()
A = np.zeros((n_atoms,n_atoms))
B = np.zeros((signal_size,n_atoms))
sparsity = []
E = []
times = []

# Model for sparse coding (alpha is the l1 penalty coefficient)
lasso = linear_model.Lasso(alpha=0.02, fit_intercept=False)
tic = time.time()
for i in tqdm(range(n_iter)):
    # Draw 1 sample at random
    rand_idx = np.random.randint(n_samples)
    y = Y[:, rand_idx].reshape((signal_size, 1))

    # Sparse coding
    x = sparse_code_lasso(y, D, lasso).reshape((50,1))
    
    # Dictionary update
    A += np.dot(x, x.T)
    B += np.dot(y, x.T)
    D = dictionary_update_omf(D, A, B)
    D = scale(D)
    
    if i%eval_interval == 0:
        # Evaluation:
        X = sparse_code_fb(Y, D, X, sparsity=4, n_iter=100)
        E.append(reconstruction_error(Y, D, X))
        times.append(time.time() - tic)
        if (time.time() - tic) > max_time:
            break
    sparsity.append(np.mean(np.sum(x!=0, axis=0)))

# Save errors and dictionary
E_omf = E.copy()
times_omf = times.copy()
D_omf = D.copy()


print('Mean sparsity: {}'.format(np.mean(sparsity)))
plt.plot(times, np.log10(E))
plt.xlabel('Time')
plt.ylabel('Error: $\log(||Y-DX||^2)$')
plt.title('Reconstruction error on the test set')
filename = 'images/omf_{}_iter_{}.png'.format(n_iter, 'synthetic' if synthetic_data else 'image')
plt.savefig(filename)
plt.show()

#plot_dictionary(D)


# # Forward Backward
# 
# Combettes, Patrick L., and Jean-Christophe Pesquet. "Proximal splitting methods in signal processing." Fixed-point algorithms for inverse problems in science and engineering. Springer New York, 2011. 185-212.  
# 
# Adapted from
# http://nbviewer.jupyter.org/github/gpeyre/numerical-tours/blob/master/matlab/sparsity_4_dictionary_learning.ipynb

# Variables
n_iter = 100
E = np.zeros(2*n_iter)
times = np.zeros(2*n_iter)
D = D0.copy()
X = X0.copy()
tic = time.time()
for i in tqdm(range(n_iter)):
    # Sparse coding
    X = sparse_code_fb(Y, D, X, sparsity=k, n_iter=100)
    E[2*i] = reconstruction_error(Y, D, X)
    times[2*i] = time.time() - tic

    # Dictionary update
    D = dictionary_update_fb(Y, D, X, n_iter=50)
    E[2*i+1] = reconstruction_error(Y, D, X)
    times[2*i+1] = time.time() - tic
    if (time.time() - tic) > max_time:
        break
# Save errors and dictionary
E_fb = E.copy()
times_fb = times.copy()
D_fb = D.copy()


plot_error(E, burn_in=2)
#plot_dictionary(D)


# ## Comparison

burn_in = 0
plt.plot(times_ksvd[burn_in:], np.log10(E_ksvd[burn_in:]), label='K-SVD')
plt.plot(times_omf[burn_in:], np.log10(E_omf[burn_in:]), label='Online matrix factorization')
plt.plot(times_fb[burn_in:], np.log10(E_fb[burn_in:]), label='Forward-Backward')
plt.xlabel('Iterations')
plt.ylabel('Error: $\log(||Y-DX||^2)$')
plt.legend()
plt.title('Reconstruction error on the test set')
filename = 'images/comparison_{}s_{}.png'.format(max_time, 'synthetic' if synthetic_data else 'image')
plt.savefig(filename)
plt.show()

