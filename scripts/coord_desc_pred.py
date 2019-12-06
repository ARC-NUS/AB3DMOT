#!/usr/bin/env python

from check_pred import get_multi_ADE
import numpy as np
from pred_obj import pred_delta_t, pred_steps, COUNT_T, label_count,PRED_MEAS_SIZE,PRED_STATE_SIZE
from coord_desc import coord_descent


def coord_desc_pred(init_params, alpha, min_alpha):
  num_params = 5
  alpha_ps = np.ones(num_params)*alpha
  
  # TODO check length of params
  is_conv, params =coord_descent(num_params=num_params, fn=get_ADE, ALPHA_PS=alpha_ps, dec_alpha=0.5, max_iter=10**3, 
                    min_alpha=min_alpha, init_params=init_params)
  return params
        
  
def get_ADE(params_v):
  parent_dir = "/media/yl/demo_ssd/raw_data"
  q_YR = params_v[0]
  q_A = params_v[1]
  R=np.eye(PRED_MEAS_SIZE) * params_v[2]
  P=np.eye(PRED_STATE_SIZE) * params_v[3]
  
  unseen_p = params_v[4]
  P[7,7]=unseen_p
  P[8,8]=unseen_p
  P[10,10]=unseen_p
  P[11,11]=unseen_p
  P[13,13]=unseen_p
  
  ADE = get_multi_ADE(parent_dir,R,P,q_YR,q_A)
  print ADE
  return -np.sum((ADE[~np.isnan(ADE)])[0:7])


if __name__ == '__main__':
  parent_dir = "/media/yl/demo_ssd/raw_data"

  R_p = 10.**-5.
  P_p = 10.**5.
  q_YR=10. ** -1.
  q_A=10. ** 50.
  unseen_p = 10. ** 50.
  
  alpha = 10.
  min_alpha = 1.
  
  R=np.eye(PRED_MEAS_SIZE) * R_p
  P=np.eye(PRED_STATE_SIZE) * P_p
  P[7,7]=unseen_p
  P[8,8]=unseen_p
  P[10,10]=unseen_p
  P[11,11]=unseen_p
  P[13,13]=unseen_p
  
  init_params = [q_YR, q_A, R_p, P_p, unseen_p]
  
  print "init score: ", get_ADE(init_params)
  
  best_params = coord_desc_pred(init_params, alpha, min_alpha)
  
  print "best", best_params
  
  q_YR=best_params[0]
  q_A=best_params[1]
  R_p = best_params[2]
  P_p = best_params[3]
  
  R=np.eye(PRED_MEAS_SIZE) * R_p
  P=np.eye(PRED_STATE_SIZE) * P_p
  
  ADE = get_multi_ADE(parent_dir,R,P,q_YR,q_A)
  print ADE
  