import numpy as np

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = (np.random.rand(inputs+1) * 2) - 1 
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x,self.bias),self.weights)
        return self.sigmoid(x_sum)

    def set_weights(self, w_init):
        """Set the weights. w_init is a python list with the weights."""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1/(1+np.exp(-x))



class MultiLayerPerceptron:     
    """A multilayer perceptron class that uses the Perceptron class above.
       Attributes:
          layers:  A python list with the number of elements per layer.
          bias:    The bias term. The same bias is used for all neurons.
          eta:     The learning rate."""

    def __init__(self, layers, bias = 1.0, eta = 0.5):
        """Return a new MLP object with the specified parameters.""" 
        self.layers = np.array(layers,dtype=object)
        self.bias = bias
        self.eta = eta
        self.network = [] # The list of lists of neurons
        self.values = []  # The list of lists of output values
        self.d = []       # The list of lists of error terms (lowercase deltas)

        for i in range(len(self.layers)):
            self.values.append([])
            self.d.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:      #network[0] is the input layer, so it has no neurons
                for j in range(self.layers[i]): 
                    self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))
        
        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values = np.array([np.array(x) for x in self.values],dtype=object)
        self.d = np.array([np.array(x) for x in self.d],dtype=object)

    def set_weights(self, w_init):
        """Set the weights. 
           w_init is a 3D list with the weights for all but the input layer."""
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])

    def print_weights(self):
        print()
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                print("Layer",i+1,"Neuron",j,self.network[i][j].weights)
        print()

    def run(self, x):
        """Feed a sample x into the MultiLayer Perceptron."""
        x = np.array(x,dtype=object)
        self.values[0] = x
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):  
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]
    
    def bp(self, x, y):
        """Run a single (x,y) pair with the backpropagation algorithm."""
        x = np.array(x,dtype=object)
        y = np.array(y,dtype=object)

        # Backpropagation Step by Step:

        # STEP 1: Feed a sample to the network 
        outputs = self.run(x)
        
        # STEP 2: Calculate the MSE
        error = (y - outputs)
        MSE = sum( error ** 2) / self.layers[-1]

        # STEP 3: Calculate the output error terms
        self.d[-1] = outputs * (1 - outputs) * (error)

        # STEP 4: Calculate the error term of each unit on each layer
        for i in reversed(range(1,len(self.network)-1)):
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i+1]): 
                    fwd_error += self.network[i+1][k].weights[h] * self.d[i+1][k]               
                self.d[i][h] = self.values[i][h] * (1-self.values[i][h]) * fwd_error

        # STEPS 5 & 6: Calculate the deltas and update the weights
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                for k in range(self.layers[i-1]+1):
                    if k==self.layers[i-1]:
                        delta = self.eta * self.d[i][j] * self.bias
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k] += delta
        return MSE



#test code
epochs = int(input("How many epochs? "))
mlp1 = MultiLayerPerceptron(layers=[7,7,1])
mlp2 = MultiLayerPerceptron(layers=[7,7,10])
mlp3 = MultiLayerPerceptron(layers=[7,7,7])

# Dataset for the 7 to 1 network
print("Training 7 to 1 network...")
for i in range(epochs):
    mse = 0.0
    mse += mlp1.bp([1,1,1,1,1,1,0],[0.05])    #0 pattern
    mse += mlp1.bp([0,1,1,0,0,0,0],[0.15])    #1 pattern
    mse += mlp1.bp([1,1,0,1,1,0,1],[0.25])    #2 pattern
    mse += mlp1.bp([1,1,1,1,0,0,1],[0.35])    #3 pattern
    mse += mlp1.bp([0,1,1,0,0,1,1],[0.45])    #4 pattern
    mse += mlp1.bp([1,0,1,1,0,1,1],[0.55])    #5 pattern
    mse += mlp1.bp([1,0,1,1,1,1,1],[0.65])    #6 pattern
    mse += mlp1.bp([1,1,1,0,0,0,0],[0.75])    #7 pattern
    mse += mlp1.bp([1,1,1,1,1,1,1],[0.85])    #8 pattern
    mse += mlp1.bp([1,1,1,1,0,1,1],[0.95])    #9 pattern
    mse = mse/10.0

# Dataset for the 7 to 10 network
print("Training 7 to 10 network...")
for i in range(epochs):
    mse = 0.0
    mse += mlp2.bp([1,1,1,1,1,1,0],[1,0,0,0,0,0,0,0,0,0])    #0 pattern
    mse += mlp2.bp([0,1,1,0,0,0,0],[0,1,0,0,0,0,0,0,0,0])    #1 pattern
    mse += mlp2.bp([1,1,0,1,1,0,1],[0,0,1,0,0,0,0,0,0,0])    #2 pattern
    mse += mlp2.bp([1,1,1,1,0,0,1],[0,0,0,1,0,0,0,0,0,0])    #3 pattern
    mse += mlp2.bp([0,1,1,0,0,1,1],[0,0,0,0,1,0,0,0,0,0])    #4 pattern
    mse += mlp2.bp([1,0,1,1,0,1,1],[0,0,0,0,0,1,0,0,0,0])    #5 pattern
    mse += mlp2.bp([1,0,1,1,1,1,1],[0,0,0,0,0,0,1,0,0,0])    #6 pattern
    mse += mlp2.bp([1,1,1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0])    #7 pattern
    mse += mlp2.bp([1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,1,0])    #8 pattern
    mse += mlp2.bp([1,1,1,1,0,1,1],[0,0,0,0,0,0,0,0,0,1])    #9 pattern
    mse = mse/10.0

# Dataset for the 7 to 7 network
print("Training 7 to 7 network...")
for i in range(epochs):
    mse = 0.0
    mse += mlp3.bp([1,1,1,1,1,1,0],[1,1,1,1,1,1,0])    #0 pattern
    mse += mlp3.bp([0,1,1,0,0,0,0],[0,1,1,0,0,0,0])    #1 pattern
    mse += mlp3.bp([1,1,0,1,1,0,1],[1,1,0,1,1,0,1])    #2 pattern
    mse += mlp3.bp([1,1,1,1,0,0,1],[1,1,1,1,0,0,1])    #3 pattern
    mse += mlp3.bp([0,1,1,0,0,1,1],[0,1,1,0,0,1,1])    #4 pattern
    mse += mlp3.bp([1,0,1,1,0,1,1],[1,0,1,1,0,1,1])    #5 pattern
    mse += mlp3.bp([1,0,1,1,1,1,1],[1,0,1,1,1,1,1])    #6 pattern
    mse += mlp3.bp([1,1,1,0,0,0,0],[1,1,1,0,0,0,0])    #7 pattern
    mse += mlp3.bp([1,1,1,1,1,1,1],[1,1,1,1,1,1,1])    #8 pattern
    mse += mlp3.bp([1,1,1,1,0,1,1],[1,1,1,1,0,1,1])    #9 pattern
    mse = mse/10.0

print("Done!\n")
pattern = [1.2]
while(pattern[0]>=0.0):
    pattern = list(map(float, input("Input pattern 'a b c d e f g': ").strip().split()))
    if pattern[0]<0.0:
        break
    print()
    print("The number recognized by the 7 to 1 network is", \
        int(mlp1.run(pattern)*10))

    print("The number recognized by the 7 to 10 network is", \
        np.argmax(mlp2.run(pattern)))

    print("The pattern recognized by the 7 to 7 network is", \
        [int(x) for x in (mlp3.run(pattern) + 0.5)] ,"\n")
