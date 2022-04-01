import torch

c = torch.tensor([])
a = torch.rand(4,1,255,255)
c = torch.cat((c, a), 0)
c = torch.cat((c, a), 0)
c = torch.cat((c, a), 0)

b = torch.squeeze(a[1].cpu()).detach().numpy()

print(b.shape)
print(c.shape)