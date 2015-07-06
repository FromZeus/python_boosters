from numba import jit, void, int_, double, autojit
import numpy as np
import time
import math

#@jit
#class test_methods():
#  @void()
#  def __init__():

t1 = time.time()
test_arr = np.arange(10000000)
t2 = time.time()
print t2 - t1
#np.random.shuffle(test_arr)

#@autojit
def test1(arr, result):
  for idx in xrange(arr.shape[0]):
    res = math.sqrt(arr[idx]) * math.exp(arr[idx] / 100000.0) / 3.0
    result[idx] = res

result = np.zeros(test_arr.shape)
#t_func = test1
t_func = jit(double[:](int_[:], double[:]))(test1)

t1 = time.time()
t_func(test_arr, result)
t2 = time.time()
print t2 - t1