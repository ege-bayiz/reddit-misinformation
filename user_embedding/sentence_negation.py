import torch
import numpy as np
from scipy.linalg import eigh
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

model_path = 'user_embedding/models/negation_mat_20231227_165538_14'
# Load the PyTorch model
model = torch.load(model_path)

print(model)
# Get the weights of the linear layer
linear_layer = model['linear.weight']

# Convert the linear layer to a numpy matrix
weights = linear_layer.cpu().numpy()

# # Perform eigenvalue decomposition
# eigenvalues, eigenvectors = eigh(weights)

# # Print the eigenvalues and eigenvectors
# print("Eigenvalues:")
# print(eigenvalues)
# print("Eigenvectors:")
# print(eigenvectors)
