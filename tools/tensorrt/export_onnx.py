"""Export the trained MS-DM density heads to a dynamic-shape ONNX model."""

import argparse
import inspect
import os

import torch
from torch import nn

import models.msdm as msdm_model


class DensityHeads(nn.Module):
    """Expose only the two density maps used during inference."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        outputs = self.model(image)
        return outputs[0], outputs[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Export MS-DM to ONNX")
    parser.add_argument(
        "--model-path", default=r"pretrained_models\msdm_final_v3_legacy.pth")
    parser.add_argument("--output", default=r"output\tensorrt\msdm.onnx")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def build_model_without_random_initialization():
    """Avoid filling parameters that the checkpoint immediately overwrites."""
    original_initializer = msdm_model._initialize_module
    original_conv_reset = nn.Conv2d.reset_parameters
    original_bn_reset = nn.BatchNorm2d.reset_parameters
    try:
        msdm_model._initialize_module = lambda module: None
        nn.Conv2d.reset_parameters = lambda module: None
        nn.BatchNorm2d.reset_parameters = lambda module: None
        return msdm_model.vgg19(pretrained=False)
    finally:
        msdm_model._initialize_module = original_initializer
        nn.Conv2d.reset_parameters = original_conv_reset
        nn.BatchNorm2d.reset_parameters = original_bn_reset


def load_checkpoint(model, path):
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = True
    checkpoint = torch.load(path, **load_kwargs)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)


def main():
    args = parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("height and width must be positive and divisible by 16")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError("checkpoint not found: {}".format(args.model_path))

    print("[ONNX] Building model...", flush=True)
    model = build_model_without_random_initialization().eval()
    load_checkpoint(model, args.model_path)
    wrapper = DensityHeads(model).eval()
    example = torch.zeros(1, 3, args.height, args.width)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("[ONNX] Exporting {}x{} model...".format(
        args.height, args.width), flush=True)
    torch.onnx.export(
        wrapper,
        example,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["whitefly_density", "fruit_fly_density"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "whitefly_density": {
                0: "batch", 2: "density_height", 3: "density_width"},
            "fruit_fly_density": {
                0: "batch", 2: "density_height", 3: "density_width"},
        },
    )

    import onnx
    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)
    print("[ONNX] Saved and validated: {}".format(output_path), flush=True)


if __name__ == "__main__":
    main()
