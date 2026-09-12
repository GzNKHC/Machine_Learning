import torch
a = torch.randn(3, 4)
print("Original:")
print(a)
print("\nFlatten:")
print(a.ravel())
print("\nReshape to 3 x 2 x 2:")
print(a.reshape(3, 2, 2))