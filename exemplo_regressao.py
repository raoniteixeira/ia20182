# Universidade Federal de Mato Grosso
# Inteligencia Artificial
# Exemplo de regressao linear utilizando o framework de otimizacao visto em sala
#
# autor: raoni at ufmt dot br

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools
from matplotlib import ticker, cm
from numpy import ma

def compute_cost(X, y, theta):
    predictions = np.matmul(X, theta)
    a = predictions - y
    sqrErrors = np.square(a)
    J = sum(sqrErrors)/(2*m)
    return J

# funcao que calcula a descida do gradiente
def gradient_descent(X, y, theta, alpha, num_iters):
    cost_history = []
    theta_history = []
    x = X[:,1].reshape(m, 1)

    for iter in range(0, num_iters):

        cost_history.append(compute_cost(X, y, theta))
        theta_history.append(np.array(theta))

        # hipotese / modelo
        h = np.matmul(X, theta)
        theta_zero = theta[0] - alpha * sum(h-y)/m
        theta_one = theta[1] - alpha *  sum(np.multiply((h-y), x))/m

        theta[0, 0] = theta_zero
        theta[1, 0] = theta_one

    return [cost_history, theta_history, theta]

data = np.loadtxt('ex1data1.txt', delimiter=',')
x = data[:,0]
m = x.size
y = data[:,1].reshape(m, 1)
x = x.reshape(m, 1)

# inclui uma coluna de 1 para facilitar a representacao vetorial da hipotese / modelo
X = np.c_[np.ones((m, 1)), x]

# vetor de parametros
theta = np.zeros((2, 1))

# parametros para descida do gradiente
num_iters = 1000
alpha = 0.01

[cost_history, theta_history, theta] = gradient_descent(X, y, theta, alpha, num_iters)

# mostra o custo em cada iteracao do algoritmo de otimizacao
plt.plot(cost_history)
plt.show()

# mostra as curvas de niveis da funcao de custo e a mudanca em theta
theta0_vals = np.arange(-10,10,.1)
theta1_vals = np.arange(-1,4,.05)
theta0, theta1 =  np.meshgrid(theta0_vals, theta1_vals)

Z = np.zeros(theta0.shape)
t = np.zeros(theta.shape)

for i in range(theta0.shape[0]):
    for j in range(theta0.shape[1]):
        t[0,0] = theta0[i,j]
        t[1,0] = theta1[i,j]
        Z[i,j] = compute_cost(X, y, t)

fig, ax = plt.subplots()
#ax = plt.axes(projection='3d')
ax.contour(theta0, theta1, Z, levels=np.arange(1,15,1), cmap='gnuplot', alpha=0.5 )
plt.plot([x[0,0] for x in theta_history], [x[1,0] for x in theta_history],'rx')
plt.show()

fig, ax = plt.subplots()
ax.contour(theta0, theta1, Z, levels=np.arange(2,15,1), cmap='gnuplot', alpha=0.50 )
plt.plot([theta[0]], [theta[1]], 'ko')
plt.show()


fig = plt.subplots()
ax = plt.axes(projection='3d')
ax.plot_surface(theta0, theta1, Z, cmap='gnuplot', alpha=0.5 )
plt.plot([x[0,0] for x in theta_history], [x[1,0] for x in theta_history], [ x[0] for x in cost_history], 'ko')
plt.show()
