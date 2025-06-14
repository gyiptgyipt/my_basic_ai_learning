import numpy as np

# Function to create a 2D rotation matrix
def rotation_matrix_2d(theta_degrees):
    theta = np.radians(theta_degrees)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

# Define the 2D vector
v = np.array([1, 0])  # along x-axis

# Define the rotation angle in degrees
angle = 90

# Create rotation matrix
R = rotation_matrix_2d(angle)

# Apply the rotation
v_rotated = R @ v
solved_v_rotated = np.round(v_rotated, decimals=2)  # Round for better readability

# Output
print("Original vector:", v)
print("Rotated vector:",v_rotated)
print(f"Rotated {angle}° counterclockwise:", solved_v_rotated)
