import torch

x = torch.rand(5, 3)

print(x)

print(torch.backends.mps.is_available()) # type: ignore
print(torch.backends.mps.is_built()) # type: ignore

print(x.device)

y = x.to(device='mps')

print(y.device)
