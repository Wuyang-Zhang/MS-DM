"""Small TensorRT 10 runner backed by CUDA tensors from PyTorch."""

import os

import numpy as np
import torch


class TensorRTRunner:
    def __init__(self, engine_path, device="cuda"):
        import tensorrt as trt

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TensorRT inference")
        if not os.path.isfile(engine_path):
            raise FileNotFoundError("TensorRT engine not found: {}".format(engine_path))
        self.trt = trt
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as stream:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(stream.read())
        if self.engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
        if self.input_names != ["input"]:
            raise RuntimeError("unexpected TensorRT inputs: {}".format(self.input_names))
        self.profile_shapes = self.engine.get_tensor_profile_shape(
            self.input_names[0], 0)
        self.stream = torch.cuda.Stream(device=self.device)

    @staticmethod
    def _torch_dtype(trt_dtype):
        numpy_dtype = np.dtype(__import__("tensorrt").nptype(trt_dtype))
        mapping = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float16): torch.float16,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.int8): torch.int8,
            np.dtype(np.bool_): torch.bool,
        }
        if numpy_dtype not in mapping:
            raise TypeError("unsupported TensorRT dtype: {}".format(trt_dtype))
        return mapping[numpy_dtype]

    def __call__(self, inputs):
        current_stream = torch.cuda.current_stream(self.device)
        inputs = inputs.to(self.device, dtype=torch.float32).contiguous()
        self.stream.wait_stream(current_stream)
        input_name = self.input_names[0]
        if not self.context.set_input_shape(input_name, tuple(inputs.shape)):
            raise ValueError(
                "input shape {} is outside the TensorRT engine profile "
                "(min={}, opt={}, max={}). A 512x512 tiled engine cannot run "
                "full images; use --inference-mode tiled/--modes tiled or "
                "build a separate --profile full engine.".format(
                    tuple(inputs.shape), *self.profile_shapes))
        self.context.set_tensor_address(input_name, inputs.data_ptr())

        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._torch_dtype(self.engine.get_tensor_dtype(name))
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            outputs[name] = tensor
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        current_stream.wait_stream(self.stream)
        whitefly = outputs["whitefly_density"]
        fruit_fly = outputs["fruit_fly_density"]
        # Match the four-item PyTorch model interface used by test.py.
        return whitefly, None, fruit_fly, None
