from grit.core import Neuron, Layer
import numpy as np
# from sklearn.datasets import make_moons, make_blobs
# X, y = make_moons(n_samples=100, noise=0.1)

w = np.array([-3., -1., 2.])
layer = Layer(w)
x = np.array([1., -2., 3.])
val = layer(x)
val.backward()
lr = 0.001
for v in layer.parameters():
    v.data -= lr * v.grad

val = layer(x)
print(val)