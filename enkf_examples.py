import pylab
from numpy.random import multivariate_normal
import numpy as np
import seaborn as sns
sns.set()

from ensemble_kalman_filter import ensemble_kf

J = 3
h = 1
gamma = np.array([[0.1, 0],
              [0, 0.1]])

x = np.array([0, 0])
cov = np.array([[1, 0],
              [0, 1]])
dim_y = 2

u_true0 = 10
u_true1 = 1
u_true = np.array([[u_true0, u_true1]] * J)
u_start = np.array([[-10, 10]] * J) + 100 * multivariate_normal(mean=x, cov=cov, size=J)

def G(u):
  return np.array([u[0], 0])

def G1(u):
  G = np.array([[3, 0],[0, 0]])
  return np.dot(G, u)

def G2(u):
  G = np.array([[2, 0],[0, 2]])
  return np.dot(G, u)

def y_observation(u, G):
  y = np.zeros((J, dim_y))
  ## noisy observation
  # G(u)
  for i in range(J):
    y[i] = G(u[i])

  # y = G(u) + eta
  eta = multivariate_normal([0] * dim_y, gamma, J)
  y += eta
  return y

y_obs = y_observation(u_true, G2)
enkf = ensemble_kf(u_start, G2, h, gamma, J ,dim_y)

u_parts0 = {}
u_parts1 = {}
u_parts2 = {}
n = 50

for i in range(n):
  enkf.step(y_obs)
  u_parts0[enkf.u_new()[0, 0]] = enkf.u_new()[0, 1]
  u_parts1[enkf.u_new()[J/2, 0]] = enkf.u_new()[J/2, 1]
  u_parts2[enkf.u_new()[J-1, 0]] = enkf.u_new()[J-1, 1]

pylab.plot(u_true0, u_true1, '*', markersize = 12)
pylab.plot(u_parts0.keys(), u_parts0.values(), '.', alpha = 0.7)
pylab.plot(u_parts1.keys(), u_parts1.values(), '.', alpha = 0.7)
pylab.plot(u_parts2.keys(), u_parts2.values(), '.', alpha = 0.7)
pylab.legend(('true particle', 'particle 0', 'particle J/2', 'particle J'))
pylab.show()
