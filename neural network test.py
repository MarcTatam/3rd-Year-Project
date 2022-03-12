import neuralnetwork as nn
import numpy as np

def relu(input):
    return np.tanh(input)

def relu_prime(input):
    return 1- np.tanh(input)**2

def xor():
    x_train = np.array([[[-1,-1]], [[-1,1]], [[1,-1]], [[1,1]]])
    y_train = np.array([[[-1]], [[1]], [[1]], [[-1]]])
    net = nn.network()
    net.add(nn.fclayer(2,3))
    net.add(nn.activation(relu,relu_prime))
    net.add(nn.fclayer(3,1))
    net.add(nn.activation(relu,relu_prime))
    net.loss_use(nn.mse,nn.mse_prime)
    net.train(x_train, y_train, epochs=1000, lr=0.1)
    out = net.predict(x_train)
    print(out)

if __name__ == "__main__":
    xor()