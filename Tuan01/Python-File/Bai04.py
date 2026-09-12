import torch
a = torch.randn(3, 4, 5)
print("Original tensor:")
print(a)
print("\nTensor a[1]:")
print(a[1])
print("\nTensor a[1:, 2:4]:")
print(a[1:, 2:4])