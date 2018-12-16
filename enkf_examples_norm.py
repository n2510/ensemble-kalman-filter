import pylab
from numpy.random import multivariate_normal
import numpy as np
import seaborn as sns
sns.set()

from ensemble_kalman_filter import ensemble_kf

J = 10
h = 0.001
gamma = np.array([[0.01, 0],
              [0, 0.01]])

x = np.array([0, 0])
cov = np.array([[0.1, 0],
              [0, 0.1]])
dim_y = 2

u_true0 = 10
u_true1 = 1
u_true = np.array([[u_true0, u_true1]] * J)
u_start = np.array([[13, 3]] * J) + multivariate_normal(mean=x, cov=cov, size=J)

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

u_list0 = []
u_list1 = []

n = 100

for i in range(n):
  enkf.step(y_obs)

  diff0 = enkf.u_new()[0] - [[u_true0, u_true1]]
  norm0 = np.linalg.norm(diff0)
  u_list0.append(norm0)

  diff1 = enkf.u_new()[J-1] - [[u_true0,u_true1]]
  norm1 = np.linalg.norm(diff1)
  u_list1.append(norm1)

zero_plot = [0] * n
pylab.plot(range(n), zero_plot,
           range(n), u_list0, '--',
           range(n), u_list1, '--')
pylab.show()
