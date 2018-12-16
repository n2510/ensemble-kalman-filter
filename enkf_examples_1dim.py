import pylab
from numpy.random import normal
import numpy as np
import seaborn as sns
sns.set()

from ensemble_kalman_filter_1dim import ensemble_kf

J = 5
h = 0.1
gamma = 0.005

mu = 0
sigma = 1

u_true0 = 0.2
u_true = [u_true0] * J
u_start = 2 + normal(mu, sigma, J)

def G(u):
  return 2 * u

def G1(u):
  return u / 2

def y_observation(u, G):
  y = [0] * J
  ## noisy observation
  # G(u)
  for i in range(J):
    y[i] = G(u[i])

  eta = normal(0, gamma, J)
  for i in range(J):
    y[i] += eta[i]
  return y

y_obs = y_observation(u_true, G1)
enkf = ensemble_kf(u_start, G1, h, gamma, J)

u_list0 = []
u_list1 = []
u_list2 = []

n = 1000

for i in range(n):
  enkf.step(y_obs)
  u_list0.append(enkf.u_new()[0])
  u_list1.append(enkf.u_new()[J/2])
  u_list2.append(enkf.u_new()[J-1])

u_true_plot = [u_true0] * n

pylab.plot(range(n), u_true_plot,
           range(n), u_list0, '--',
           range(n), u_list1, '--',
           range(n), u_list2, '--')
pylab.legend(('true particle', 'particle 0', 'particle J/2', 'particle J'))
pylab.show()
