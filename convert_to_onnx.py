import torch
import tensorrt as trt
import torch.nn.functional as F
import numpy as np
import cv2
from imread_from_url import imread_from_url

from nets import Model

import pdb

if __name__ == '__main__':

	model_path = "models/crestereo_eth3d.pth"

	model = Model(max_disp=256, mixed_precision=False, test_mode=True)
	model.load_state_dict(torch.load(model_path), strict=True)
	model = model.eval().to("cuda")
	
	pdb.set_trace()

	in_h, in_w = (720, 1280)
	t1_half = torch.rand(1, 3, in_h//2, in_w//2)
	t2_half = torch.rand(1, 3, in_h//2, in_w//2)

	t1 = torch.rand(1, 3, in_h, in_w)
	t2 = torch.rand(1, 3, in_h, in_w)
	flow_init = torch.rand(1, 2, in_h//2, in_w//2)

	# Export the model
	torch.onnx.export(model,               
	                  (t1, t2, flow_init),
	                  "crestereo.onnx",   # where to save the model (can be a file or file-like object)
	                  export_params=True,        # store the trained parameter weights inside the model file
	                  opset_version=12,          # the ONNX version to export the model to
	                  do_constant_folding=True,  # whether to execute constant folding for optimization
	                  input_names = ['left', 'right','flow_init'],   # the model's input names
	                  output_names = ['output'])

	# Export the model without init_flow (it takes a lot of time)
	# !! Does not work prior to pytorch 1.12 (confirmed working on pytorch 2.0.0)
	# Ref: https://github.com/pytorch/pytorch/pull/73760
	torch.onnx.export(model,               
	                  (t1_half, t2_half),
	                  "crestereo_without_flow.onnx",   # where to save the model (can be a file or file-like object)
	                  export_params=True,        # store the trained parameter weights inside the model file
	                  opset_version=12,          # the ONNX version to export the model to
	                  do_constant_folding=True,  # whether to execute constant folding for optimization
	                  input_names = ['left', 'right'],   # the model's input names
	                  output_names = ['output'])

	pdb.set_trace()

	model_without_flow_onnx = onnx.load("models/crestereo_without_flow.onnx")
	model_with_flow_onnx = onnx.load("models/crestereo.onnx")
	
	# Create a TensorRT builder and network
	builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
	network = builder.create_network()
        
        # Create an ONNX-TensorRT backend
	parser = trt.OnnxParser(network, builder.logger)
	parser.parse(model_without_flow_onnx.SerializeToString())
	
	# Set up optimization profile and builder parameters
	profile = builder.create_optimization_profile()
	profile.set_shape("input", (1, 10), (1, 10), (1, 10))
	builder_config = builder.create_builder_config()
	builder_config.max_workspace_size = 1 << 30
	builder_config.flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)


