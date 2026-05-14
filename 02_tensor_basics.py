import torch 

A = torch.tensor([
    [1, 2],
    [3, 4]
])

B = torch.tensor([
    [5, 6],
    [7, 8]
])

print(torch.matmul(A, B))

# range of values from 0 to 11
x = torch.arange(12)
print(x)

# moving tensor to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor([1, 2, 3]).to(device)
print(x)

# Autograd and gradients
# Autograd automatically calculates gradients.
# Gradients help neural networks learn.
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
print(y)

# Compute gradients
y.backward() # computes the gradient of y with respect to x: at x=2, dy/dx = 2*2 + 3 = 7
print(x.grad)