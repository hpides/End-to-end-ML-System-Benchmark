import torch

print("Matrix Multiplying")
# Create random matrices and move them to the GPU
a = torch.rand((20000, 20000), device='cpu')
b = torch.rand((20000, 20000), device='cpu')

# Perform matrix multiplication on the GPU
ab = torch.matmul(a, b)

# Move result back to CPU and print (optional, here we'll print only a small part of it)
ab_cpu = ab.cpu()
print(ab_cpu[:5, :5])