cimport cython
import numpy as np
cimport numpy as np
import time

from libc.math cimport sqrt
from libc.math cimport exp

t1 = time.time()
test_arr = np.arange(10000000)
t2 = time.time()
print t2 - t1


cdef test1(np.ndarray[np.int64_t, ndim = 1] arr,
  np.ndarray[np.float64_t, ndim = 1] result):
  cdef int idx
  cdef float res
  for idx in range(arr.shape[0]):
    res = sqrt(arr[idx]) * exp(arr[idx] / 100000.0) / 3.0
    result[idx] = res


result = np.zeros(test_arr.shape)

t1 = time.time()
test1(test_arr, result)
t2 = time.time()
print t2 - t1