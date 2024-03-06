from grit.core import Neuron, Layer
import numpy as np

w = np.array([-3, -1, 2])
layer = Layer(w)
x = np.array([1, -2, 3])
val = layer(x)
val.backward()
# neuron = Neuron(w)
# for w in neuron.w:
#     w.backward()

print(val)