#feature1
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Generate data
x = np.linspace(-5, 5, 100)
y_sigmoid = sigmoid(x)


# Plot graphs
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.legend()


plt.tight_layout()
plt.show()


