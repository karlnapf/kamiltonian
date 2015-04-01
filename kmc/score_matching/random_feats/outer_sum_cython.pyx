cimport numpy as np
cimport cython
import numpy as np

@cython.boundscheck(False)
def outer_sum_cython(np.ndarray[double, ndim=2] X):
    cdef unsigned int i, j, k
    cdef np.ndarray C = np.zeros([X.shape[1], X.shape[1]], dtype=np.float64)
    
    for n in range(X.shape[0]):
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                C[i,j] += X[n, i] * X[n, j]
    
    return C
