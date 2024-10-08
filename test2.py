
import torch

def scalar_transform(x: torch.Tensor):
    """ Reference from MuZerp: Appendix F => Network Architecture
    & Appendix A : Proposition A.2 in https://arxiv.org/pdf/1805.11593.pdf (Page-11)
    """
    assert len(x.shape) == 1
    epsilon = 0.001
    sign = torch.ones(x.shape).float()
    sign[x < 0] = -1.0
    output = sign * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x
    return output


a = torch.tensor([-30.0, -20.0, -10.0, 5])
b = scalar_transform(a)
print(b)