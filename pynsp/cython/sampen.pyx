cimport cython
from libcpp.vector cimport vector
from libc.math cimport log as log_c
from libc.math cimport sqrt as sqrt_c
import numpy as np
cimport numpy as np
# import cython
from cython.parallel import prange, parallel


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double abs_c(double n) nogil:
    if n < 0:
        return n * -1
    else:
        return n


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double STD(double[:] data) nogil:
    cdef long n = data.shape[0]
    cdef long i
    cdef double avr = 0
    cdef double stdv = 0

    for i in range(n):
        avr += data[i]
    avr = avr / n

    for i in range(n):
        stdv += (data[i] - avr) ** 2
    return sqrt_c(stdv / n)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double MAX(double a, double b) nogil:
    if a > b:
        return a
    else:
        return b


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double SampEn(double[:] xvec, char w, double r) nogil:
    cdef long nval = xvec.shape[0]
    cdef double A = 0
    cdef double B = 0
    cdef double distA, distB
    cdef double R = r * STD(xvec)
    cdef long ii, jj, kk

    for ii in range(nval - w):
        for jj in range(nval - w):
            if ii == jj:
                pass
            else:
                distA = 0.0
                distB = 0.0
                for kk in range(w + 1):
                    if (nval - w - ii >= 0) or (nval - w - jj) >= 0:
                        distB = MAX(distB, abs_c(xvec[ii + kk] - xvec[jj + kk]))
                    if kk == w + 1:
                        pass
                    else:
                        distA = MAX(distA, abs_c(xvec[ii + kk] - xvec[jj + kk]))
                if distA < R:
                    A += 1
                if distB < R:
                    B += 1
    if (A > 0) and (B > 0):
        dist = log_c(A / B)
    else:
        dist = 0.0
    return dist


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray SampEn3D(np.ndarray[np.double_t, ndim = 4] U,
                          char m, double r, int n_thread):
    # Use Memoryviews type
    cdef double[:, :, :, :] U_view = U
    cdef int x, y, z, i, j, k
    cdef double result
    x = U.shape[0]
    y = U.shape[1]
    z = U.shape[2]

    # defind Memoryviews for output
    cdef double[:, :, :] R_view = np.empty([x, y, z], dtype=np.double)
    with nogil, parallel(num_threads=n_thread):
        for i in prange(x):
            for j in range(y):
                for k in range(z):
                    R_view[i, j, k] = SampEn(U_view[i, j, k, :], m, r)
    return np.asarray(R_view)