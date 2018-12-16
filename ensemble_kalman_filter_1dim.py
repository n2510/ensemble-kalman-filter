import numpy as np
from numpy.random import normal

class ensemble_kf:

  def __init__(self, _u, _G, _h, _gamma, _J):
    self.u = _u               # unknown parameters
    self.G = _G               # response operator
    self.h = _h               # step-size
    self.gamma = _gamma       # covariance of noise
    self.J = _J               # number of particles
    self.y = [0] * self.J     # artificial data

  def u_new(self):
    return self.u

  ### add new artificial data y to enkf
  def step(self, y_observation):
    J = self.J
    u_current = self.u    # u_n
    # print 'u_current:', u_current.round(2)

    G_operator = [0] * self.J
    for i in range(J):
      G_operator[i] = self.G(u_current[i])
    # print '* G_op *:', [ round(elem, 2) for elem in G_operator ]

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
    # print 'C_up', C_up.round(2)

    ## artificial data
    xi_cov = self.gamma / self.h
    xi = normal(0, xi_cov, J)

    for i in range(J):
      self.y[i] = y_observation[i] + xi[i]

    # print 'y_obs:', [ round(elem, 2) for elem in y_observation ]
    # print '* y *:', [ round(elem, 2) for elem in self.y ]

    ## kalman gain
    K = C_up / (C_pp + xi_cov)
    # print 'K:', K.round(2)

    ## u_n+1 - jth particle
    for i in range(J):
      self.u[i] +=  K * (self.y[i] - G_operator[i])
    # print '** u_n+1 ** :', [ round(elem, 2) for elem in self.u ]
    # print '---------'
