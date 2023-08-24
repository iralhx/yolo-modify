import math
import  torch
from typing import Optional, Tuple
from torch import nn, Tensor
from ultralytics.nn.modules.block import C2f_DCN,Conv
from torch.autograd import Function
from torchvision.ops.deform_conv import deform_conv2d

class DeformConv2d(Function):
    """
    See :func:`deform_conv2d`.
    """

    @staticmethod
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,kernel_size)
        self.stride = (stride,stride)
        self.padding = (padding,padding)
        self.dilation = (dilation,dilation)
        self.groups = groups

        self.weight =nn.Parameter(
            torch.empty(out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1])
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @staticmethod
    def forward(self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
            offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
                offsets to be applied for each position in the convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
                masks to be applied for each position in the convolution kernel.
        """
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

# class DCNv2(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=1, dilation=1, groups=1, deformable_groups=1):
#         super(DCNv2, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = (kernel_size, kernel_size)
#         self.stride = (stride, stride)
#         self.padding = (padding, padding)
#         self.dilation = (dilation, dilation)
#         self.groups = groups
#         self.deformable_groups = deformable_groups

#         self.weight = nn.Parameter(
#             torch.empty(out_channels, in_channels, *self.kernel_size)
#         )
#         self.bias = nn.Parameter(torch.empty(out_channels))

#         out_channels_offset_mask = (self.deformable_groups * 3 *
#                                     self.kernel_size[0] * self.kernel_size[1])
#         self.conv_offset_mask = nn.Conv2d(
#             self.in_channels,
#             out_channels_offset_mask,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             bias=True,
#         )
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = Conv.default_act
#         self.reset_parameters()

#     def forward(self, x):
#         offset_mask = self.conv_offset_mask(x)
#         o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#         x = torch.ops.torchvision.deform_conv2d(
#             x,
#             self.weight,
#             offset,
#             mask,
#             self.bias,
#             self.stride[0], self.stride[1],
#             self.padding[0], self.padding[1],
#             self.dilation[0], self.dilation[1],
#             self.groups,
#             self.deformable_groups,
#             True
#         )
#         x = self.bn(x)
#         x = self.act(x)
#         return x

#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         std = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-std, std)
#         self.bias.data.zero_()
#         self.conv_offset_mask.weight.data.zero_()
#         self.conv_offset_mask.bias.data.zero_()

# def deform_conv2d_onnx(g, input,
#         weight,
#         offset,
#         mask,
#         bias,
#         stride_h,
#         stride_w,
#         pad_h,
#         pad_w,
#         dil_h,
#         dil_w,
#         n_weight_grps,
#         n_offset_grps,
#         use_mask,)-> Tensor:
#     return g.op("torchvision::deform_conv2d", input,
#         weight,
#         offset,
#         mask,
#         bias,
#         stride_h,
#         stride_w,
#         pad_h,
#         pad_w,
#         dil_h,
#         dil_w,
#         n_weight_grps,
#         n_offset_grps,
#         use_mask,)

# from torch.onnx import register_custom_op_symbolic
# register_custom_op_symbolic('torchvision::deform_conv2d', deform_conv2d_onnx, 9)


class DevNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c2f=DeformConv2d(in_channels= 1,out_channels= 3,kernel_size= 3)
    def forward(self,x):
        x=self.c2f(x)
        return x
