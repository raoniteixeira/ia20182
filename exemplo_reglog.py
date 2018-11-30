# Universidade Federal de Mato Grosso
# Inteligencia Artificial
# Exemplo de regressao logistica utilizando o framework de otimizacao visto em sala
#
# autor: raoni at ufmt dot br

import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1 + np.exp(-z));

def gradient(theta,x,y):
    m , n = x.shape
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    sigmoid_x_theta = sigmoid(x.dot(theta));
    grad = ((x.T).dot(sigmoid_x_theta-y))/m;
    return grad.flatten();

def compute_cost(theta,x,y):
    m,n = x.shape; 
    z = np.matmul(X, theta)	
    term1 = np.log(sigmoid(z));
    term2 = np.log(1-sigmoid(z));
    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))
    term = y * term1 + (1 - y) * term2;
    J = -((np.sum(term))/m);
    return J;


data = np.loadtxt('ex2data1.txt', delimiter=',')
x = data[:,0:2]
m = x.shape[0]
y = data[:,2].reshape(m, 1)


# inclui uma coluna de 1 para facilitar a representacao vetorial da hipotese / modelo
X = np.c_[np.ones((m, 1)), x]

# vetor de parametros
theta = np.zeros((3, 1))

Result = op.minimize(fun = compute_cost, x0 = theta, args = (X, y), method = 'TNC', jac = gradient);
optimal_theta = Result.x;

# mostrando os dados
plt.plot(x[np.where(y > 0), 0], x[np.where(y > 0), 1], 'ko')
plt.plot(x[np.where(y == 0), 0], x[np.where(y == 0), 1], 'ro')

# mostrando a superficie de decisao
w = optimal_theta
a = -w[2] / w[1]
xx = np.linspace(30, 100)
yy = a * xx - (optimal_theta[0]) / w[1]
plt.plot(xx, yy, 'b-')

plt.show()
