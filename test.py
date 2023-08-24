from ultralytics import YOLO
from model_test import *
import torch
import torch.onnx




def deform_conv2d_onnx(g, input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,)-> Tensor:
    return g.op("torchvision::deform_conv2d", input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,)

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic('torchvision::deform_conv2d', deform_conv2d_onnx, 9)


# Load a model
model = YOLO('yolov8dcn.yaml')  # build a new model from YAML


model.export(format='onnx')

