# inspiration from micrograd https://github.com/karpathy/micrograd
import numpy as np

np.random.seed(1337)

class Value:

    def __init__(self, data, _children=()):

        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)

    def __str__(self):

        return "Value %r grad %r" % (self.data, self.grad)
    
    # Add magic method
    # Return the current data plus the other data
    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        
        return out
    
    # Multiply magic method
    # Return the current data multiplied by the other data
    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward

        return out
    
    # Rectified linear unit activation function
    # If the value is less than 0, return 0
    # Otherwise return the data
    def relu(self):

        out = Value(0 if self.data < 0 else self.data, (self,))

        # The derivative of the ReLU is 1 if the value is greater than 0 and 0 otherwise
        # Chain rule means to multiply this with the output grad
        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out
    
    # Main backward method
    # Build using topographic sort
    # E.g., start with current state, then go to all the children and their children
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            # Check if node in visited
            if v not in visited:
                visited.add(v)
                # Iterate through all children
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Set the gradient to 1 by default
        self.grad = 1

        # Reverse the list to start with the latest function
        for v in reversed(topo):
            # Call the backwards function on that node
            v._backward()
            

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class Neuron(Module):

    # Initialize with random weights based on number of inputs
    # Set the bias to be 0
    def __init__(self, data=None, nInputs=3):

        if isinstance(data, np.ndarray):
            self.w = [Value(val) for val in data]
        else:
            self.w = [Value(np.random.uniform(-1, 1)) for _ in range(nInputs)]

        self.b = Value(1)   # Initialized to 0 for example; TODO: change back to 0 later

    def __str__(self):

        return "Neuron (weights: %r) (bias: %r)" % (self.numpy(self.w), self.numpy([self.b]))
    
    # When the neuron is called with inputs x, perform the forward pass
    # i.e., w1 * x1 + w2 * x2 + ... + wn * xn
    def __call__(self, x):

        out = np.sum([wi*xi for wi,xi in zip(self.w, x)] + [self.b])
        return out.relu()
    
    def parameters(self):
        return self.w + [self.b]
    
    # Static method to print the numpy array being used
    @staticmethod
    def numpy(arr):

        return np.array([a.data for a in arr])
    
class Layer(Module):

    # Initialize with data weights
    def __init__(self, data=None, nInputs=3, nOutputs=1):

        if isinstance(data, np.ndarray):
            self.data = data
            self._reshape()
            self.neurons = [Neuron(row) for row in self.data]
            print(self.neurons[0])
        else:
            self.neurons = [Neuron(nInputs) for _ in range(nOutputs)]
        

    def __str__(self):

        return "Layer (%r)" % ([n.numpy(n.w) for n in self.neurons])
    
    # When the layer is called, each neuron should multiple the incoming vector
    def __call__(self, x):

        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    # Reshape check for 1D tensors
    def _reshape(self):

        if self.data.ndim == 1:
            self.data = np.reshape(self.data, (-1, self.data.shape[0]))

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]