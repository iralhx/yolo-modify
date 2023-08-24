import torch
import torch.nn as nn
from torch.autograd import Function
import onnx
import torch.onnx

from torch.onnx import register_custom_op_symbolic


def my_group_norm(g, input, num_groups, scale, bias, eps):
    return g.op("mydomain::mygroupnorm", input, num_groups, scale, bias, epsilon_f=eps)


register_custom_op_symbolic('mynamespace::custom_group_norm', my_group_norm, 9)


class Requant_(Function):
    @staticmethod
    def forward(ctx, input, requant_scale, shift):               # ctx 必须要
        input = input.double() * requant_scale / 2**shift        # 为了等价于c中的移位操作。会存在int32溢出
        input = torch.floor(input).float()

        return torch.floor(input)
    
    @staticmethod
    def symbolic(g, *inputs):
        return g.op("Requant", inputs[0], scale_f=23.0, shift_i=8)

requant_ = Requant_.apply

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(-1)
        x = requant_(x, 5, 5)
        return x

net = TinyNet().cuda()
ipt = torch.ones(2,3,12,12).cuda()
torch.onnx.export(net, (ipt,), 'tinynet.onnx', opset_version=11)
print(onnx.load('tinynet.onnx'))
