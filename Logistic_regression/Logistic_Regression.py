import numpy as np

def activation(z):
  return 1/(1 + np.exp(-z))

def gradient_descent(x, y, theta,alpha, m, h1):
    grad = np.dot((h1 - y), x) / m
    theta = theta - alpha*grad
    return theta

def loss(x, y, theta, m,h):
  loss1 = np.dot(np.log(h),y)
  loss2 = np.dot((np.log(1-h)), (1-y))
  loss_out = -1*(loss1+ loss2)/m
  return loss_out
