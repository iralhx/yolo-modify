from ultralytics import YOLO

# Load a model
model = YOLO('yolov8dcn.yaml')  # build a new model from YAML


model.export(format='onnx')

