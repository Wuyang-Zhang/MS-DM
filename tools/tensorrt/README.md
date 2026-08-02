# TensorRT Acceleration

This directory contains the complete ONNX and TensorRT acceleration pipeline
for MS-DM. Run every command from the repository root.

## Environment

The pipeline has been tested with the existing `bisenet` Conda environment:

```text
Python        3.10
PyTorch       2.4.1 + CUDA 12.4
ONNX          1.21
TensorRT      10.10
GPU           NVIDIA GeForce RTX 4060 Laptop GPU
```

Activate it before using these tools:

```powershell
conda activate bisenet
```

TensorRT engines are tied to the TensorRT version, GPU architecture, precision,
and optimization profile. Build the engine on the machine where it will run.

## Files

| File | Purpose |
| --- | --- |
| `export_onnx.py` | Loads the `.pth` checkpoint and exports the two density heads to dynamic-shape ONNX. |
| `build_tensorrt.py` | Uses `trtexec` to build an FP16 or FP32 TensorRT engine. |
| `runtime.py` | Runs TensorRT 10 engines with CUDA tensors supplied by PyTorch. |
| `validate.py` | Compares TensorRT FP16 outputs and latency against PyTorch FP32. |

The scripts write generated files to `output/tensorrt/` by default:

```text
output/tensorrt/
|-- msdm.onnx
`-- msdm_tiled_fp16.engine
```

The `output/` directory is ignored by Git. ONNX files and TensorRT engines are
generated artifacts and should not be committed.

## Recommended Order

### 1. Export ONNX

```powershell
python -m tools.tensorrt.export_onnx
```

The exporter loads `pretrained_models\msdm_final_v3_legacy.pth`, exports a
dynamic batch/height/width model, and runs the ONNX checker.

Optional paths and export dimensions can be supplied explicitly:

```powershell
python -m tools.tensorrt.export_onnx `
  --model-path pretrained_models\msdm_final_v3_legacy.pth `
  --output output\tensorrt\msdm.onnx `
  --height 512 `
  --width 512
```

### 2. Build a Tiled FP16 Engine

The recommended engine accepts `512 x 512` tiles and dynamic batch sizes 1--4:

```powershell
python -m tools.tensorrt.build_tensorrt `
  --profile tiled `
  --tile-size 512 `
  --max-batch-size 4
```

Engine construction normally takes several minutes because TensorRT benchmarks
multiple implementation tactics. Use `--fp32` only when FP16 is unsuitable.

### 3. Validate Accuracy and Speed

```powershell
python -m tools.tensorrt.validate
```

The validator uses a real image tile and reports maximum/mean density error,
predicted-count error, PyTorch latency, TensorRT latency, and acceleration.

The validated RTX 4060 Laptop result for one `512 x 512` tile was:

```text
PyTorch FP32       25.488 ms   39.23 FPS
TensorRT FP16       5.284 ms  189.26 FPS
Speedup                         4.82x
Whitefly count error            0.1783%
Fruit-fly count error           0.0129%
```

### 4. Predict with TensorRT

TensorRT uses the normal prediction entry point, so output visualization,
density maps, positions, counts, and CSV summaries remain unchanged:

```powershell
python predict.py `
  --backend tensorrt `
  --engine-path output\tensorrt\msdm_tiled_fp16.engine `
  --inference-mode tiled `
  --tile-size 512 `
  --tile-overlap 64 `
  --tile-batch-size 4
```

The tile size must match the tiled engine profile, and the requested batch size
must not exceed the engine's `--max-batch-size`.
When `--backend tensorrt` is selected, `predict.py` defaults to tiled inference.

### CUDA Graph

CUDA Graph is enabled by default for the TensorRT backend. For every input
shape encountered (for example, batches 1 and 4 of `3 x 512 x 512` tiles), the
runtime allocates stable input/output buffers, performs one TensorRT warm-up,
and captures the engine execution. Later calls with the same shape copy into
the cached input buffer and replay the captured graph. This reduces CPU kernel
launch overhead without changing model outputs.

The first call for each new batch shape includes allocation, warm-up, and graph
capture time. Keep warm-up runs enabled when measuring FPS. To disable CUDA
Graph for debugging or comparison, add:

```powershell
--disable-cuda-graph
```

On the tested RTX 4060 Laptop GPU, one `512 x 512` tile measured approximately
`5.501 ms` without CUDA Graph and `5.270 ms` with it, a further latency
reduction of about 4.2%. The benefit is expected to be smaller than the main
FP16 TensorRT conversion because CUDA Graph only reduces launch overhead.

### 5. Benchmark TensorRT

```powershell
python benchmark_fps.py `
  --backend tensorrt `
  --engine-path output\tensorrt\msdm_tiled_fp16.engine `
  --modes tiled `
  --warmup-runs 2 `
  --benchmark-runs 10 `
  --tile-size 512 `
  --tile-overlap 64 `
  --tile-batch-size 4
```

When `--backend tensorrt` is selected, the benchmark defaults to
`--modes tiled`. PyTorch continues to default to both full and tiled modes.

For the included `1440 x 1920` example with 20 tiles, a short validation run
with CUDA Graph and GPU stitching measured approximately 9.65 model-forward
images/s and 8.86 overall images/s. Overall throughput also includes tile
preparation and the final density transfer to CPU.

Tiled density accumulation, overlap weighting, and normalization are performed
on the GPU. Per-batch outputs are not converted immediately with
`.cpu().numpy()`; the two completed species maps are stacked and transferred to
CPU together after all tiles have been stitched. This removes one GPU
synchronization and two device-to-host transfers per tile batch.

## Full-Image Profile

The default engine is optimized only for tiled inference. To support dynamic
full-image input, build a separate engine:

```powershell
python -m tools.tensorrt.build_tensorrt `
  --profile full `
  --engine output\tensorrt\msdm_full_fp16.engine `
  --min-size 384 `
  --opt-height 1440 `
  --opt-width 1920 `
  --max-height 1920 `
  --max-width 1920
```

Then use `--inference-mode full` and pass that engine to `predict.py`. Building
and running a full-image engine requires substantially more GPU memory than the
tiled profile.

## Common Problems

- `trtexec was not found on PATH`: add the TensorRT `bin` directory to `PATH`.
- `input shape is outside the engine profile`: rebuild the engine with a shape
  and batch profile that includes the requested input.
- Engine deserialization failure: rebuild it using the installed TensorRT
  version and the target GPU.
- Out-of-memory error: reduce tile batch size, reduce the full-image maximum
  dimensions, or use tiled inference.
