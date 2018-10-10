import torch

data = torch.Tensor([[1, 2], [3, 4], [5, 6]])
label = torch.LongTensor([0, 0, 1])
label_ = torch.Tensor([0, 0, 1])
center = torch.Tensor([[1, 1], [2, 2]])

# data = torch.Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [1, 10]])
# label = torch.LongTensor([1, 0, 1, 3, 3])
# label_ = torch.Tensor([1, 0, 1, 3, 3])
# label_ = torch.Tensor([0, 1, 1, 3, 3])
# center = torch.Tensor([[1, 1], [2, 2], [3, 3], [4, 4]])

center_exp = center.index_select(dim=0, index=label)
# print(center_exp)
#
count = torch.histc(label_, bins=10, min=0, max=10)
ccc = count.index_select(dim=0, index=label)
print(ccc)
#
a = (data - center_exp) ** 2
print(a)
b = torch.sum(a, 1)
print(b)
c = torch.sqrt(b)
print(c)
d = c / ccc
print(d)
f = torch.sum(d)
print(f)
