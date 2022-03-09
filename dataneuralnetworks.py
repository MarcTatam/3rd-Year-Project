import neuralnetwork as nn
import numpy as np

def relu(input):
    return np.maximum(input,0)

def relu_prime_single(item):
    if item > 0 :
        return 1
    elif item < 0:
        return 0

def relu_prime(input):
    return np.array(list(map(relu_prime_single, input)))



def cell_network():
    net = nn.network()
    net.add(nn.fclayer(24,16))
    net.add(nn.activation(relu, relu_prime))
    net.add(nn.fclayer(16,1))
    net.add(nn.activation(relu, relu_prime))

if __name__ == "__main__":
    temp = np.arange(-10,10)
    print(relu_prime(temp))