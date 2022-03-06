# Q1_graded
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower
    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T
    return X, Y

def plot_decision_boundary(model, X, y, name, index):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.subplot(1, 2, index)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    # print(X[0, :].shape, X[1, :].shape, y.shape)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.title(name)
    


X1, Y1 = load_planar_dataset()


def do_train(X, Y, a=0.1, n=100, Type='batch'):

    m = Y.shape[1]    # number of samples
    #change data shape
    Y = Y.ravel() 
    X = X.T
    b = np.ones((m,1))
    X = np.concatenate((X, b), axis=1)

    I = list(range(m))
    W = np.array([1,1,0.])

    if Type == 'batch':
        for _ in range(n):
            Sum = np.zeros(3)
            for i in I:
                e = Y[i] - np.dot(X[i], W)
                Sum += np.dot(X[i], e*a)
            W +=  (Sum/m) 
            if _ % 100 == 0:
                print(f'batch:  iteration= {_} \t error= {e}')

    elif Type == '‫‪stochastic‬‬':
        for _ in range(n):
            np.random.shuffle(I)
            for i in I:
                e = Y[i] - np.dot(X[i], W)
                W += np.dot(X[i], e*a)
            if _ % 100 == 0:
                print(f'‫‪stochastic‬‬:  iteration= {_} \t error= {e}')

    return W


def my_model(inputs):
    l = []
    for inp in inputs:     
        y = inp[0] * W[0] + inp[1] * W[1] + W[2]
        l.append(1 if y<=0 else 0)
    return np.array(l)


print('########  start batch #####################')
W = do_train(X1, Y1, a=0.0001, n=2000, Type='batch')
plot_decision_boundary(my_model, X1, Y1,name='batch', index=1)

print('########  start ‫‪stochastic‬‬ #####################')
W = do_train(X1, Y1, a=0.000001, n=1000, Type='‫‪stochastic‬‬')
plot_decision_boundary(my_model, X1, Y1,name='‫‪stochastic‬‬', index=2)



