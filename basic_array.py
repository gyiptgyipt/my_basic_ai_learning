import numpy as np

# Create arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise addition
c = a + b
print("a + b =", c)

# Dot product
dot_product = np.dot(a, b)
print("Dot product =", dot_product)

# Reshape array
matrix = np.arange(1, 7).reshape(2, 3)
print("Reshaped matrix:\n", matrix)

# Mean and standard deviation
print("Mean:", np.mean(matrix))
print("Standard Deviation:", np.std(matrix))
