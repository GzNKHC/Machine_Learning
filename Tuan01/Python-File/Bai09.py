import numpy as np
import torch
polynomial = np.poly1d([1, 2, 3])
N = 20
X = np.random.randn(N, 1) * 5
Y = polynomial(X)
XX = np.hstack([
    X * X,
    X,
    np.ones_like(X)
])
w = torch.randn(3, 1, requires_grad=True)
x = torch.tensor(XX, dtype=torch.float32)
y = torch.tensor(Y, dtype=torch.float32)
optimizer = torch.optim.NAdam([w], lr=0.01)
print("Initial coefficients:")
print(w)
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = x @ w
    mse = torch.mean(torch.square(y - y_pred))
    mse.backward()
    optimizer.step()
print("Learned coefficients:")
print(w)