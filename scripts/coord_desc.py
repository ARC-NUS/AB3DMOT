#!/usr/bin/env python
# coding: utf-8

import numpy as np


def x_sq(x):
  return -(x[0]-4)**2 +10.-x[1]**2


'''
@brief: coordinate descent method in the paper "Discriminative Training of Kalman Filter"
@param num_params: number of parameters to search for. Must be INT
@param fn: function call. needs to take in the parameters as numpy array. 
           Must return single value which indicates the "goodness" (moar bigger moar better)
@param ALPHA_PS: the numpy array of the alphas for each param
@param max_iter: maximum number of iteration before stopping the optimisation
@param min_alpha: early stop if all the alphas are smaller than min_alpha
@param init_params: initial guess of the params in 1D-numpy array. optional
@param param_range: range of param for the init param if it is randomly sampled
@outputs: true if it converges and false otherwise
'''
def coord_descent(num_params, fn, ALPHA_PS, dec_alpha, max_iter, min_alpha, init_params=None, param_range=10.**3, fn_params=None):
  if init_params is None:
    params = (np.random.rand(num_params) - 0.5 ) * param_range
  else:
    params = init_params
  # TODO check num_params
  # TODO check if numpy is valid by checking size

  # temporary variables
  if fn_params is None:
    best_result = fn(params)
  else:
    # print "fn params:", fn_params
    fn_full = (params,) + fn_params
    # print "fn params:", fn_full
    best_result = fn(*fn_full)
  print  "init", best_result, params
  is_conv = False
  alpha_ps = ALPHA_PS

  # choose param/direction
  for desc_iter in range(max_iter):
    # print "is conv", is_conv
    if np.amax(alpha_ps) <= min_alpha:
      print "converge at desc_iter", desc_iter
      is_conv = True
      break
    # print "is conv", is_conv
    # alpha_ps = ALPHA_PS
    for i_param in range(num_params):
      is_moved = False
      while not is_moved:
        # set alphas =1. for all the other params
        alpha_p = np.ones(num_params)
        alpha_p[i_param] = alpha_ps[i_param]

        # test increase
        p_test = (alpha_p) * params
        if fn_params is None:
          test_result = fn(p_test)
        else:
          fn_full = (p_test,) + fn_params
          test_result = fn(*fn_full)
          # print "fn params:", fn_full
        # print "inc",i_param, p_test, "res", test_result
        # print test_result > best_result
        if test_result > best_result:
          params = p_test
          best_result=test_result
          # print "best", best_result, params
          is_moved = True
        else:
          # test decrease # FIXME should compare between the inc and dec 
          p_test = params / alpha_p
          if fn_params is None:
            test_result = fn(p_test)
          else:
            fn_full = (p_test,) + fn_params
            test_result = fn(*fn_full)
            # print "fn params:", fn_full
          # print "dec", i_param,p_test, "res", test_result
          if test_result > best_result:
            params = p_test
            best_result=test_result
            print "best", best_result, params
            is_moved = True
          else:
            # reduce alpha
            alpha_ps[i_param] *= dec_alpha
            # print "reduce alpha ", i_param, "to ", alpha_ps[i_param]
            if abs(alpha_ps[i_param]) <= min_alpha:
              # print "could not find better alpha"
              break
  return is_conv, params
  
if __name__ == '__main__':
  num_params=2
  is_conv, params =coord_descent(num_params=num_params, fn=x_sq, ALPHA_PS=np.ones(num_params)*10., dec_alpha=0.5, max_iter=10**3, 
                min_alpha=10.**-10, init_params=[1.,3.])
  print "converged?", is_conv
  print "best params", params
  print "best score", x_sq(params)
