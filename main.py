#feature2
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions


def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Generate data
x = np.linspace(-5, 5, 100)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)

# Plot graphs
plt.figure(figsize=(10, 6))



plt.subplot(2, 2, 2)
plt.plot(x, y_relu, label='ReLU', color='orange')
plt.title('ReLU Activation Function')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='green')
plt.title('Leaky ReLU Activation Function')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, y_tanh, label='Tanh', color='red')
plt.title('Tanh Activation Function')
plt.legend()

plt.tight_layout()
plt.show()