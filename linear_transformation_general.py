import numpy as np

# Define a 3x3 transformation matrix (e.g., scaling and shearing)
A = np.array([[2, 0, 0],
              [0, 1, 1],
              [0, -1, 1]])

# Define a vector in R³
v = np.array([1, 2, 3])

# Apply the transformation
result = A @ v # use the @ operator for matrix multiplication(what is different from the dot product)
print("Original vector:", v)
print("Transformation matrix:\n", A)
print("Transformed vector:", result)
