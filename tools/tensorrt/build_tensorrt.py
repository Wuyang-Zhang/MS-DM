"""Build an FP16 TensorRT engine from the exported MS-DM ONNX model."""

import argparse
import os
import shutil
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Build an MS-DM TensorRT engine")
    parser.add_argument("--onnx", default=r"output\tensorrt\msdm.onnx")
    parser.add_argument("--engine", default=r"output\tensorrt\msdm_tiled_fp16.engine")
    parser.add_argument("--profile", choices=("tiled", "full"), default="tiled")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--min-size", type=int, default=384)
    parser.add_argument("--opt-height", type=int, default=1440)
    parser.add_argument("--opt-width", type=int, default=1920)
    parser.add_argument("--max-height", type=int, default=1920)
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--fp32", action="store_true")
    return parser.parse_args()


def shape_profiles(args):
    if args.profile == "tiled":
        size = args.tile_size
        return (
            "input:1x3x{}x{}".format(size, size),
            "input:{}x3x{}x{}".format(
                min(2, args.max_batch_size), size, size),
            "input:{}x3x{}x{}".format(args.max_batch_size, size, size),
        )
    return (
        "input:1x3x{0}x{0}".format(args.min_size),
        "input:1x3x{}x{}".format(args.opt_height, args.opt_width),
        "input:1x3x{}x{}".format(args.max_height, args.max_width),
    )


def main():
    args = parse_args()
    if not os.path.isfile(args.onnx):
        raise FileNotFoundError("ONNX model not found: {}".format(args.onnx))
    if args.max_batch_size <= 0:
        raise ValueError("max batch size must be positive")
    trtexec = shutil.which("trtexec")
    if not trtexec:
        raise FileNotFoundError("trtexec was not found on PATH")

    engine_path = os.path.abspath(args.engine)
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    minimum, optimum, maximum = shape_profiles(args)
    command = [
        trtexec,
        "--onnx={}".format(os.path.abspath(args.onnx)),
        "--saveEngine={}".format(engine_path),
        "--minShapes={}".format(minimum),
        "--optShapes={}".format(optimum),
        "--maxShapes={}".format(maximum),
        # trtexec interprets the workspace value in MiB; adding a textual
        # suffix makes recent TensorRT versions treat it as a byte count.
        "--memPoolSize=workspace:{}".format(args.workspace_mib),
        "--skipInference",
    ]
    if not args.fp32:
        command.append("--fp16")
    print("[TensorRT] Profile: {} / {} / {}".format(
        minimum, optimum, maximum), flush=True)
    subprocess.run(command, check=True)
    print("[TensorRT] Saved engine: {}".format(engine_path), flush=True)


if __name__ == "__main__":
    main()
