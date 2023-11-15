import torch
import torch.nn.functional as F

import torch.nn
import torch.optim
import torch.profiler
import torch.utils.data
import torchvision.datasets
import torchvision.models
import torchvision.transforms as T
import torch.autograd.profiler as profiler

import torch.nn.utils.prune as prune
import torch_tensorrt

import time
import numpy as np
import cv2
from imread_from_url import imread_from_url

import pdb

from nets import Model

device = 'cuda'

# cudnn.benchmark = True
def dataPreparation(left, right):
	imgL = left.transpose(2, 0, 1)
	imgR = right.transpose(2, 0, 1)
	imgL = np.ascontiguousarray(imgL[None, :, :, :])
	imgR = np.ascontiguousarray(imgR[None, :, :, :])

	imgL = torch.tensor(imgL.astype("float32")).to(device)
	imgR = torch.tensor(imgR.astype("float32")).to(device)

	imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)

	return imgL, imgR, imgL_dw2, imgR_dw2
	

def benchmark(model, left, right, n_iter=20, nwarmup=50, nruns=1000):

	print("Model Forwarding...")
	imgL, imgR, imgL_dw2, imgR_dw2 = dataPreparation(left, right)

	left_shape = imgL.shape
	right_shape = imgR.shape

	left_data = torch.randn(left_shape)
	left_data = left_data.to("cuda")
	right_data = torch.randn(right_shape)
	right_data = right_data.to("cuda")


	left_dw2_shape = imgL_dw2.shape
	right_dw2_shape = imgR_dw2.shape
	left_dw2_data = torch.randn(left_dw2_shape)
	left_dw2_data = left_dw2_data.to("cuda")
	right_dw2_data = torch.randn(right_dw2_shape)
	right_dw2_data = right_dw2_data.to("cuda")

	# if dtype=='fp16':
	#     input_data = input_data.half()

	# pdb.set_trace()
	    
	print("Warm up ...")
	with torch.no_grad():
	    for _ in range(nwarmup):
	        pred_flow_dw2_warmup = model(left_dw2_data, right_dw2_data, iters=n_iter, flow_init=None)
	        pred_flow = model(left_data, right_data, iters=n_iter, flow_init=pred_flow_dw2_warmup)

	torch.cuda.synchronize()

	print("Start timing ...")
	# pdb.set_trace()
	timings = []
	with torch.no_grad():
	    for i in range(1, nruns+1):
	        start_time = time.time()
	        pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
	        pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
	        torch.cuda.synchronize()
	        end_time = time.time()
	        timings.append(end_time - start_time)
	        print('BENCHMARK: Iteration %d/%d, avg time %.2f ms'%(i, nruns, np.mean(timings)*1000))



#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, n_iter=20):

	print("Model Forwarding...")
	imgL, imgR, imgL_dw2, imgR_dw2 = dataPreparation(left, right)

	# pdb.set_trace()

	print("Warm up ...")
	with torch.no_grad():
	    for _ in range(10):
	        pred_flow_dw2_warmup = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
	        pred_flow_warmup = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2_warmup)

	torch.cuda.synchronize()


	# with torch.autograd.profiler.profile(use_cuda=True) as prof:


	timings = []
	with profiler.profile(with_stack=True, use_cuda=True) as prof:
		start_time = time.time()
		with torch.inference_mode():
			
		    pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
		    pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
		    
		    # timings.append(end_time - start_time)

		end_time = time.time()
		timings.append(end_time - start_time)
		pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

	# print('INFERENCE: time %.2f ms'%((timings[0])*1000))

	# prof.stop()
	print(prof)

	return pred_disp


if __name__ == '__main__':

	left_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
	right_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")

	in_h, in_w = left_img.shape[:2]

	# Resize image in case the GPU memory overflows
	eval_h, eval_w = (in_h,in_w)
	assert eval_h%8 == 0, "input height should be divisible by 8"
	assert eval_w%8 == 0, "input width should be divisible by 8"
	
	imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
	imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

	model_path = "models/crestereo_eth3d.pth"

	model = Model(max_disp=256, mixed_precision=False, test_mode=True)
	model.load_state_dict(torch.load(model_path), strict=True)
	model.to(device)
	model.eval()

	# TensorRT
	imgL_rt, imgR_rt, imgL_dw2_rt, imgR_dw2_rt = dataPreparation(imgL, imgR)

	a = torch.randn(imgL_rt.shape).to("cuda")
	b = torch.randn(imgR_rt.shape).to("cuda")
	c = torch.randn(imgL_dw2_rt.shape).to("cuda")
	d = torch.randn(imgR_dw2_rt.shape).to("cuda")

	
	# def trace_wrpper(a, b, c, d):
	# 	pdb.set_trace();
	# 	return model(a, b, c, d, 20, None);


	# m = model(c, d, iters=20, flow_init=None)
	# traced_model = torch.jit.trace(model, (a, b, m, torch.tensor(20)))


	# trt_input = [
	# 	torch_tensorrt.Input([1, 3, 720, 1280]),
	# 	torch_tensorrt.Input([1, 3, 720, 1280]),
	# 	torch_tensorrt.Input([1, 3, 360, 640]),
	# 	torch_tensorrt.Input([1]),
	# ]

	# pdb.set_trace()
	# trt_model = torch_tensorrt.compile(model,
	# 									    inputs= trt_input,
	# 									    enabled_precisions= { torch_tensorrt.dtype.half} # Run with FP16
	# 									)

	benchmark(model, imgL, imgR, n_iter=20, nwarmup=5, nruns=10)
	# pdb.set_trace()
	pred = inference(imgL, imgR, model, n_iter=20)

	t = float(in_w) / float(eval_w)
	disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

	disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
	disp_vis = disp_vis.astype("uint8")
	disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
	
	# pdb.set_trace()
	cv2.imwrite("left.png", left_img)
	cv2.imwrite("right.png", right_img)
	cv2.imwrite("disp.png", disp_vis)

	# combined_img = np.hstack((left_img, disp_vis))
	# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
	# cv2.imshow("output", combined_img)
	# cv2.imwrite("output.jpg", disp_vis)
	# cv2.waitKey(0)



