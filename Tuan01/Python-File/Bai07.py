import torch
a = torch.randn(3, 4)
print(a)
print("\nMean:")
print(torch.mean(a, dim=0))
print("\nStandard deviation:")
print(torch.std(a, dim=0))
print("\nCumulative sum:")
print(torch.cumsum(a, dim=0))