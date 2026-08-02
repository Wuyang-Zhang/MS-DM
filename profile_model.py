"""Report MS-DM parameter size and approximate floating-point operations."""

import argparse
import inspect
import os
from collections import defaultdict

import torch
from torch import nn

import models.msdm as msdm_model


def parse_args():
    parser = argparse.ArgumentParser(description="Profile MS-DM model complexity")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device", default="cpu",
        help="profiling device (CPU is the device-independent default)",
    )
    parser.add_argument(
        "--model-path", default=r"pretrained_models\msdm_final_v3_legacy.pth",
        help="optional checkpoint used to report file size and validate loading",
    )
    parser.add_argument(
        "--skip-checkpoint", action="store_true",
        help="profile the architecture without loading a checkpoint",
    )
    return parser.parse_args()


def resolve_device(requested):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def load_checkpoint(model, path, device):
    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    checkpoint = torch.load(path, **load_kwargs)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)


def build_model_for_profiling(device):
    """Build the architecture without its redundant random reinitialization.

    Profiling does not use random parameter values, and a requested checkpoint
    overwrites every parameter immediately afterwards. Skipping the historical
    whole-model initialization therefore keeps counts identical while avoiding
    several minutes of unnecessary startup work.
    """
    original_initializer = msdm_model._initialize_module
    original_conv_reset = nn.Conv2d.reset_parameters
    original_bn_reset = nn.BatchNorm2d.reset_parameters
    try:
        msdm_model._initialize_module = lambda module: None
        nn.Conv2d.reset_parameters = lambda module: None
        nn.BatchNorm2d.reset_parameters = lambda module: None
        return msdm_model.vgg19(pretrained=False).to(device).eval()
    finally:
        msdm_model._initialize_module = original_initializer
        nn.Conv2d.reset_parameters = original_conv_reset
        nn.BatchNorm2d.reset_parameters = original_bn_reset


def tensor_bytes(tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def human_count(value):
    for suffix, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if value >= scale:
            return "{:.3f} {}".format(value / scale, suffix)
    return str(int(value))


def human_bytes(value):
    for suffix, scale in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2),
                          ("KiB", 1024)):
        if value >= scale:
            return "{:.3f} {}".format(value / scale, suffix)
    return "{} B".format(int(value))


class OperationCounter:
    """Count MACs and common element-wise FLOPs through forward hooks."""

    def __init__(self):
        self.macs = defaultdict(int)
        self.elementwise_flops = defaultdict(int)
        self.handles = []

    def register(self, model):
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_hook(self._conv2d))
            elif isinstance(module, nn.ConvTranspose2d):
                self.handles.append(module.register_forward_hook(self._conv_transpose2d))
            elif isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear))
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.handles.append(module.register_forward_hook(self._batch_norm))
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Sigmoid)):
                self.handles.append(module.register_forward_hook(self._activation))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _conv2d(self, module, inputs, output):
        kernel_operations = (
            module.kernel_size[0] * module.kernel_size[1]
            * module.in_channels // module.groups
        )
        self.macs["Conv2d"] += output.numel() * kernel_operations
        if module.bias is not None:
            self.elementwise_flops["Conv2d bias"] += output.numel()

    def _conv_transpose2d(self, module, inputs, output):
        kernel_operations = (
            module.kernel_size[0] * module.kernel_size[1]
            * module.in_channels // module.groups
        )
        self.macs["ConvTranspose2d"] += output.numel() * kernel_operations
        if module.bias is not None:
            self.elementwise_flops["ConvTranspose2d bias"] += output.numel()

    def _linear(self, module, inputs, output):
        output_elements = output.numel()
        self.macs["Linear"] += output_elements * module.in_features
        if module.bias is not None:
            self.elementwise_flops["Linear bias"] += output_elements

    def _batch_norm(self, module, inputs, output):
        # Inference-time affine scale and shift.
        self.elementwise_flops["BatchNorm"] += output.numel() * 2

    def _activation(self, module, inputs, output):
        self.elementwise_flops[type(module).__name__] += output.numel()


def profile_operations(model, inputs):
    counter = OperationCounter()
    counter.register(model)
    with torch.no_grad():
        outputs = model(inputs)
    counter.remove()
    return counter, outputs


def print_table(counter):
    print("\nOperation breakdown")
    print("-------------------")
    for name, macs in sorted(counter.macs.items()):
        print("{:<24} MACs {:>14} | FLOPs {:>14}".format(
            name, human_count(macs), human_count(2 * macs)))
    for name, flops in sorted(counter.elementwise_flops.items()):
        print("{:<24} FLOPs {:>14}".format(name, human_count(flops)))


def main():
    args = parse_args()
    if args.height <= 0 or args.width <= 0 or args.batch_size <= 0:
        raise ValueError("height, width, and batch size must be positive")
    if args.height % 8 or args.width % 8:
        raise ValueError("height and width must be divisible by 8")

    device = resolve_device(args.device)
    print("[PROFILE] Building MS-DM on {}...".format(device), flush=True)
    model = build_model_for_profiling(device)
    if not args.skip_checkpoint:
        if not os.path.isfile(args.model_path):
            raise FileNotFoundError("checkpoint not found: {}".format(args.model_path))
        print("[PROFILE] Loading checkpoint: {}".format(args.model_path), flush=True)
        load_checkpoint(model, args.model_path, device)

    parameters = list(model.parameters())
    buffers = list(model.buffers())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    trainable_count = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad)
    parameter_storage = tensor_bytes(parameters)
    state_storage = parameter_storage + tensor_bytes(buffers)

    inputs = torch.zeros(
        args.batch_size, 3, args.height, args.width, device=device)
    print("[PROFILE] Running one profiling forward pass...", flush=True)
    counter, outputs = profile_operations(model, inputs)
    total_macs = sum(counter.macs.values())
    total_elementwise = sum(counter.elementwise_flops.values())
    total_flops = 2 * total_macs + total_elementwise

    print("MS-DM model profile")
    print("====================")
    print("Device:              {}".format(device))
    print("Input:               {} x 3 x {} x {}".format(
        args.batch_size, args.height, args.width))
    print("Output shapes:       {}".format(
        [tuple(output.shape) for output in outputs]))
    print("Parameters:          {} ({:,})".format(
        human_count(parameter_count), parameter_count))
    print("Trainable parameters:{} ({:,})".format(
        " " + human_count(trainable_count), trainable_count))
    print("Parameter FP32 size: {}".format(human_bytes(parameter_count * 4)))
    print("Parameter FP16 size: {}".format(human_bytes(parameter_count * 2)))
    print("Current param storage:{}".format(" " + human_bytes(parameter_storage)))
    print("State dict storage:  {}".format(human_bytes(state_storage)))
    if not args.skip_checkpoint:
        print("Checkpoint file:     {}".format(
            human_bytes(os.path.getsize(args.model_path))))
    print("MACs:                {} ({:,})".format(
        human_count(total_macs), total_macs))
    print("Approximate FLOPs:   {} ({:,})".format(
        human_count(total_flops), total_flops))
    print_table(counter)
    print("\nNotes")
    print("-----")
    print("- One multiply-accumulate (MAC) is reported as two FLOPs.")
    print("- Counts include Conv2d, ConvTranspose2d, Linear, BatchNorm, and activations.")
    print("- Functional interpolation, concatenation, pooling, indexing, and additions")
    print("  are not included, so FLOPs are an architecture-level approximation.")
    print("- FLOPs scale with batch size and spatial input dimensions.")


if __name__ == "__main__":
    main()
