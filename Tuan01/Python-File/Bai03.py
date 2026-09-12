import torch
a = torch.zeros(2, 3, 4)
print("Tensor:")
print(a)
print("Shape:", a.shape)
print("Size:", a.size())
print("Number of dimensions:", a.ndim)
print("Data type:", a.dtype)