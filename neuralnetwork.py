import numpy as np

class layer:
    """Base class for a layer of the neural network"""

    def __init__(self):
        self.input = None
        self.output = None

    def forward_propo(self,input):
        """Computes the output of a layer given an input
        
        Args
        Input - input to the layer"""
        raise NotImplementedError

    def back_propo(self, output_err, lr):
        """Calculates the partial derivative with respect to this layer
        
        Args 
        output_err - error of the output
        lr - The learning rate"""
        raise NotImplementedError

class fclayer(layer):
    """Fully connected layer of neurons"""
    def __init__(self, in_size, out_size):
        """Constructor
        
        Args
        in_size - number of input neurons
        output_size - number of output neurons"""
        #Adjacency Matrix for weights
        self.weights = np.random.rand(in_size, out_size) - 0.5
        #Bias Matrix
        self.bias = np.random.rand(1, out_size)

    def forward_propo(self,input):
        """Computes the output of a layer given an input
        
        Args
        Input - input to the layer"""
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output

    def back_propo(self, output_err, lr):
        """Calculates the partial derivative with respect to this layer
        
        Args 
        output_err - error of the output
        lr - The learning rate
        
        Returns
        Input error of the layer"""
        #With respect to weights
        input_err = np.dot(output_err, self.weights.T)
        weights_err = np.dot(self.input.T, output_err)

        #Update param
        self.weights -= lr * weights_err
        self.bias -= lr * output_err
        return input_err

class activation(layer):
    """Activation Layer"""
    def __init__(self, activation, activation_prime):
        """Constructor
        
        Args
        activation - activation function
        activation_prime - derivative of activation function"""
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propo(self,input):
        """Computes the output of a layer given an input
        
        Args
        Input - input to the layer"""
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def back_propo(self, output_err,lr):
        """Calculates the partial derivative with respect to this layer
        
        Args 
        output_err - error of the output"""
        return self.activation_prime(self.input)*output_err

def mse(actual, predicted):
    """Calcualtes the mean square error
    
    Args
    actual - actual values list
    predicted - predicted values list
    
    Returns
    Mean squared error"""
    return np.mean(np.power(actual-predicted,2))

def mse_prime(actual, predicted):
    """Calculates the derivative of the mean squared error
    
    Args
    actual - actual values list
    predicted - predicted values list
    
    Returns
    Derivative of mean squared error"""
    return 2*(predicted-actual)/actual.size

class network:
    """Neural Network class"""

    def __init__(self):
        """Constructor"""
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, new_layer):
        """Adds a layer to the network
        
        Args
        new_layer - layer to be added"""
        self.layers.append(new_layer)

    def loss_use(self, loss, loss_prime):
        """Sets the loss that should be used
        
        Args
        loss - The loss value
        loss_prime - Derivative of the loss value"""
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, inputs):
        """Predicts the output for a given set of inputs
        
        Args
        inputs - input data
        
        Return
        Predicted values"""
        sample_count = len(inputs)
        results = []

        for i in range(sample_count):
            output = inputs[i]
            for network_layer in self.layers:
                output = network_layer.forward_propo(output)
            results.append(output)
        return results

    def train(self, train_inputs, train_outputs, epochs, lr):
        """Trains the neural network
        
        Args
        train_inputs - training data input
        train_outputs - Expected outputs
        epochs - Number of times to adjust weights
        lr - learning rate"""
        sample_count = len(train_inputs)

        #Learning Loop
        for i in range(epochs):
            err = 0
            for j in range(sample_count):
                # forward propagation
                output = train_inputs[j]
                for network_layer in self.layers:
                    output = network_layer.forward_propo(output)

                # compute loss (for display purpose only)
                err += self.loss(train_outputs[j], output)

                # backward propagation
                error = self.loss_prime(train_outputs[j], output)
                for network_layer in reversed(self.layers):
                    error = network_layer.back_propo(error, lr)
            err /= sample_count
            print('Epoch %d of %d   error=%f' % (i+1, epochs, err))
if __name__ == "__main__":
    this_lay = fclayer()