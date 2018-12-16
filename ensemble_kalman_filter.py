import numpy as np
from numpy.random import multivariate_normal

class ensemble_kf:

  def __init__(self, _u, _G, _h, _gamma, _J, _dim_y):
    self.u = _u               # unknown parameters
    self.G = _G               # response operator
    self.h = _h               # step-size
    self.gamma = _gamma       # covariance of noise
    self.J = _J               # number of particles
    self.dim_y = _dim_y       # dim y

  def u_new(self):
    return self.u

  ### add new artificial data y to kalman filter
  def step(self, y_observation):
    J = self.J
    u_current = self.u    # u_n
    # print 'u_current:', u_current.round(2)
    inv = np.linalg.inv

    G_operator = np.zeros((J, self.dim_y))
    for i in range(J):
      G_operator[i] = self.G(u_current[i])
    # print '* G_op *:', G_operator.round(2)

    ## empirical means
    G_mean = np.mean(G_operator, axis = 0)
    u_mean = np.mean(u_current, axis = 0)
    # print 'G_mean:', G_mean.round(2)
    # print 'u_mean:', u_mean.round(2)

    ## empirical covariances
    # C_pp
    C_pp = 0
    for G_op in G_operator:
      G_error = G_op - G_mean
      # print 'G_error:', G_error
      C_pp += np.tensordot(G_error, G_error, 0)
    C_pp /= J

    # C_up
    C_up = 0
    for i in range(J):
      u_error = u_current[i] - u_mean
      G_error = G_operator[i] - G_mean
      C_up += np.tensordot(u_error, G_error, 0)
    C_up /= J

    # print 'C_pp:', C_pp.round(2)
    # print 'C_up:', C_up.round(2)

    ## artificial data
    xi_cov = self.gamma / self.h
    # print 'xi_cov:', xi_cov
    xi = multivariate_normal([0] * self.dim_y, xi_cov, J)
    self.y = y_observation + xi

    # print 'y_obs:', y_observation.round(2)
    # print '* y *:', self.y.round(2)

    ## kalman gain
    K = np.dot(C_up, inv(C_pp + xi_cov))
    # print 'K:', K.round(2)

    ## u_n+1 - jth particle
    for i in range(J):
      self.u[i] +=  np.dot(K, self.y[i] - G_operator[i])

    # print '** u_n+1 ** :', self.u.round(2)
    # print '---------'
