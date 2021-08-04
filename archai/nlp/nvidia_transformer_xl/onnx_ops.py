import numpy as np

import torch

torch_triu = torch.triu
def triu_onnx(x, diagonal=0):
    assert len(x.shape) == 2
    arange = torch.arange(x.size(0), device = x.device)
    arange2 = torch.arange(x.size(1), device = x.device)
    mask = arange.unsqueeze(-1).expand(-1, x.size(1)) <= (arange2 - diagonal)
    return x.masked_fill(mask==0, 0)
torch.triu = triu_onnx

# for _ in range(1000):
#     dim1 = np.random.randint(1, 500)
#     dim2 = np.random.randint(1, 500)
#     diag = np.random.randint(0, min(dim1, dim2))
#     print(dim1, dim2, diag)
#     a = torch.randn(dim1, dim2)
#     gt = torch_triu(a, diagonal=diag)
#     ours = triu_onnx(a, diagonal=diag)
#     assert (gt==ours).all()

torch_outer = torch.outer
def outer_onnx(input, vec2):
    input = input.unsqueeze(-1)
    vec2 = vec2.unsqueeze(0)
    return torch.matmul(input, vec2)
torch.outer = outer_onnx

for _ in range(1000):
    dim1 = np.random.randint(1, 500)
    dim2 = np.random.randint(1, 500)
    print(dim1, dim2)
    a = torch.randn(dim1)
    b = torch.randn(dim2)
    gt = torch_outer(a, b)
    ours = outer_onnx(a, b)
    assert (gt==ours).all()