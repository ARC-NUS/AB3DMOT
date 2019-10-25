#!/usr/bin/env python
# coding: utf-8

def x_sq(x):
  return x[0]**2


'''
@brief: coordinate descent method in the paper "Discriminative Training of Kalman Filter"
@param num_params: number of parameters to search for. Must be INT
@param fn: function call. needs to take in the parameters as numpy array
@param alpha_ps: the 
@param max_iter
@param min_alpha
@param init_params: initial guess of the params in 1D-numpy array. optional
@outputs: true if it converges and false otherwise
'''
def coord_descent(num_params, fn, alpha_ps, max_iter, min_alpha, init_params=None):
  if init_params is None:
    init_params = np.random.rand(num_params)
    
  
  params = init_params
  old_results = np.inf
  # TODO check num_params
  # TODO check if numpy is valid by checking size
  
  is_conv = False
  
  # test increase
  p_test = alpha_ps * init_params
  test_results =fn(p_test)
  if test_results > old_results:
    params = p_test
    
    
  
  
  return is_conv
  
if __name__ == '__main__':
  coord_descent(1, x_sq, 10., 100, 0.01, init_params=[10.])