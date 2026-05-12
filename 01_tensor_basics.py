import torch 

x = torch.tensor([[1, 2, 3]])
print(x)
print(x.dtype)
print(x.shape)
print(x.device)

print("Is CUDA available?", torch.cuda.is_available())

print("*" * 20)
# scalar tensor 
# x = torch.tensor(3)
# x = torch.empty(3) # 1D vector
# x = torch.zeros(2, 3) # 2D matrix
# x = torch.ones(3)
# x = torch.empty(3, 4) # 2D matrix
# tensor([  [0., 0., 0., 0.], 
#           [0., 0., 0., 0.], 
#           [0., 0., 0., 0.] 
# ])
# x = torch.empty(2, 3, 4) # 3D tensor
# x = torch.empty(2, 2, 3, 4) # 4D tensor
# x = torch.rand(3, 4, dtype=torch.float32, device=torch.device('cpu')) # random values between 0 and 1
# y = torch.rand(3, 4, dtype=torch.float32, device=torch.device('cpu')) # random values between 0 and 1

x = torch.rand(3, 4, dtype=torch.float32)
y = torch.rand(3, 4, dtype=torch.float32)

# x = x.to("cuda") # move tensor to GPU
# y = y.to("cuda") # move tensor to GPU
# z = x + y  # happens inside the GPU

# use built-in arithmetic functions 
z = torch.add(x, y) # happens inside the GPU   x + y
# z = torch.mul(x, y) # happens inside the GPU   x * y
# z = torch.div(x, y) # happens inside the GPU   x / y
# z = torch.sub(x, y) # happens inside the GPU   x - y

# move z back to CPU for printing
# zz = z.to("cpu")
# print(zz)

print(x.dtype)
print(x.shape)
print(x.device)

print("*" * 20)
x = torch.rand(4, 4, dtype=torch.float32, device=torch.device('cpu')) # random values between 0 and 1
print(x)
print(x.dtype)
print(x.shape)
print(x.device)

# y = x.view(16) # reshape to 1D vector
# y = x.reshape(16) # reshape to 1D vector
# y = x.reshape(8, 2) # reshape to 2D matrix
# y = x.view(-1, 2) # reshape to 2D matrix, 
y = x.view(8, -1) # reshape to 2D matrix, 
# -1 means infer the size of that dimension based on the other dimensions and the total number of elements
print(y)
print(y.dtype)
print(y.shape)
print(y.device)

print("*" * 20)
# convert torch tensor to numpy array and vice versa
x = torch.rand(3, 4, dtype=torch.float32) # random values between 0 and 1
y = x.numpy() # convert to numpy array
print(y)
print(type(y))

# convert numpy array to torch tensor
z = torch.from_numpy(y) # convert back to torch tensor
print(z)
print(z.dtype)

print("*" * 20)
# slicing 
x = torch.rand(5, 3, dtype=torch.float32) # random values between 0 and 1
print(x)
# print(x[0]) # first row
print(x[:, 1]) # all rows, column 1
print(x[0, :]) # row 0, all columns
print(x[1,1]) # row 1, column 1
print(x[0,0].item()) # get the value as a Python number
