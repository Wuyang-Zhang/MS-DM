"""Compare TensorRT FP16 outputs and latency against the PyTorch model."""

import argparse
import time

import cv2
import torch

from .export_onnx import build_model_without_random_initialization, load_checkpoint
from predict import image_tensor
from .runtime import TensorRTRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Validate MS-DM TensorRT output")
    parser.add_argument("--image", default=r"data\predict\IMG_20221230_135744.jpg")
    parser.add_argument(
        "--model-path", default=r"pretrained_models\msdm_final_v3_legacy.pth")
    parser.add_argument(
        "--engine-path", default=r"output\tensorrt\msdm_tiled_fp16.engine")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-runs", type=int, default=20)
    return parser.parse_args()


def latency_ms(call, inputs, warmups, runs):
    for _ in range(warmups):
        call(inputs)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(runs):
        call(inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / runs


def main():
    args = parse_args()
    source = cv2.imread(args.image)
    if source is None:
        raise FileNotFoundError("image could not be read: {}".format(args.image))
    inputs = image_tensor(source)
    if inputs.shape[2] < args.tile_size or inputs.shape[3] < args.tile_size:
        raise ValueError("validation image is smaller than the requested tile")
    inputs = inputs[:, :, :args.tile_size, :args.tile_size].cuda().contiguous()

    pytorch_model = build_model_without_random_initialization()
    load_checkpoint(pytorch_model, args.model_path)
    pytorch_model = pytorch_model.cuda().eval()
    tensorrt_model = TensorRTRunner(args.engine_path)

    def pytorch_call(value):
        with torch.no_grad():
            outputs = pytorch_model(value)
        return outputs[0], outputs[2]

    def tensorrt_call(value):
        outputs = tensorrt_model(value)
        return outputs[0], outputs[2]

    pytorch_outputs = pytorch_call(inputs)
    tensorrt_outputs = tensorrt_call(inputs)
    torch.cuda.synchronize()
    for name, reference, candidate in zip(
            ("whitefly", "fruit_fly"), pytorch_outputs, tensorrt_outputs):
        difference = (reference - candidate).abs()
        reference_count = reference.sum().item()
        candidate_count = candidate.sum().item()
        relative_count_error = abs(candidate_count - reference_count) / max(
            abs(reference_count), 1e-12)
        print(
            "[ACCURACY] {} | max abs {:.6g} | mean abs {:.6g} | "
            "count {:.6f} -> {:.6f} | count error {:.4%}".format(
                name, difference.max().item(), difference.mean().item(),
                reference_count, candidate_count, relative_count_error),
            flush=True,
        )

    pytorch_ms = latency_ms(
        pytorch_call, inputs, args.warmup_runs, args.benchmark_runs)
    tensorrt_ms = latency_ms(
        tensorrt_call, inputs, args.warmup_runs, args.benchmark_runs)
    print("[LATENCY] PyTorch: {:.3f} ms ({:.2f} FPS)".format(
        pytorch_ms, 1000.0 / pytorch_ms), flush=True)
    print("[LATENCY] TensorRT: {:.3f} ms ({:.2f} FPS)".format(
        tensorrt_ms, 1000.0 / tensorrt_ms), flush=True)
    print("[SPEEDUP] {:.2f}x".format(pytorch_ms / tensorrt_ms), flush=True)


if __name__ == "__main__":
    main()
