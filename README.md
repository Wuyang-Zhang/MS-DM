# MS-DM

MS-DM 是论文 **A multi-species pest recognition and counting method based on a density map in the greenhouse** 的 PyTorch 实现，用于对白粉虱（whitefly）和果蝇（fruit fly）进行双物种计数与位置可视化。

- 论文：*Computers and Electronics in Agriculture*, 217 (2024), 108554
- DOI：[10.1016/j.compag.2023.108554](https://doi.org/10.1016/j.compag.2023.108554)
- 基础方法：[DM-Count, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/118bd558033a1016fcc82560c65cca5f-Abstract.html)
- 移动端项目：[MS_DM_Android](https://github.com/Wuyang-Zhang/MS_DM_Android)

## 方法概览

本项目以 DM-Count 的分布匹配计数框架为基础，主要扩展包括：

- 使用两套点标注和两个回归分支，分别预测白粉虱与果蝇密度图；
- 使用 VGG19 作为共享特征提取骨干；
- 使用 FPN 融合高层语义信息与低层细节；
- 在果蝇分支加入由 ASPP 和 CBAM 组成的 Context-and-Focus（ConF）模块；
- 使用计数损失、Optimal Transport（OT）损失和 Total Variation（TV）损失训练；
- 支持将预测密度图转换为原图上的位置结果；
- 使用自适应样本均衡方法缓解两个物种的数量不平衡。

最终模型入口为：

```text
models/models_aspp_cmba_conv_ff_fusion.py
```

`from models import vgg19` 默认导入该最终结构。

## 项目结构

```text
MS-DM/
├── data/                         # 训练与验证数据
├── datasets/                     # 数据集读取和数据增强
├── losses/                       # OT、Sinkhorn 等损失实现
├── models/                       # MS-DM 网络结构
├── tools/                        # 数据预处理工具
├── pretrained_models/            # 预训练权重（默认被 Git 忽略）
├── ckpts/                        # 训练检查点和日志（默认被 Git 忽略）
├── train.py                      # 训练入口
├── train_helper.py               # 训练与验证流程
└── test.py                        # 统一的计数、密度图与位置预测入口
```

## 环境

### 原始复现环境

论文代码的原始依赖为：

- Python 3.7
- PyTorch 1.2.0
- TorchVision 0.4.0
- CUDA 10.0

本机已有 Conda 环境：

```powershell
conda activate msdm
```

该环境适合检查旧代码和权重兼容性。但 PyTorch 1.2/CUDA 10 对 RTX 40 系列显卡支持较差，在 RTX 4060 上训练和推理可能异常缓慢。

### RTX 40 系列建议环境

实际推理建议使用支持当前显卡架构的较新环境，例如：

- PyTorch 2.x
- CUDA 12.x
- TorchVision 与 PyTorch 对应版本

本机已验证 `bisenet` 环境中的 PyTorch 2.4.1 + CUDA 12.4 可以运行两个测试脚本：

```powershell
conda activate bisenet
```

这是本机现有环境名称，并非项目强制要求。其他机器建议单独创建现代 PyTorch 环境。

现代环境安装依赖：

```powershell
pip install -r requirements.txt
```

复现 Python 3.7/PyTorch 1.2 环境时使用：

```powershell
pip install -r requirements-legacy.txt
```

## 数据目录

两个物种使用并行目录，文件名必须对应：

```text
data/
├── data-used-by-train-val-test/
│   ├── train/
│   ├── val/
│   └── test/
└── data-used-by-train-val-test-another/
    ├── train/
    ├── val/
    └── test/
```

每张图片使用同名 `.npy` 保存点坐标：

```text
0001.jpg
0001.npy
```

当前代码约定：

- `data-used-by-train-val-test`：第一个计数分支的标注；
- `data-used-by-train-val-test-another`：第二个计数分支的标注；
- 点坐标格式为 `(x, y)`；
- 训练裁剪尺寸默认为 512，输出密度图为输入的 1/8 尺度。

常用预处理脚本位于 `data/` 和 `tools/preprocess/`。运行前应根据实际路径检查脚本参数，不建议直接依赖脚本中的历史绝对路径。

## 权重

论文最终结构对应的现有最佳权重为：

```text
ckpts/input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0-v3 fusion/best_model_16.pth
```

该权重与最终网络的 92 个参数项完全匹配。由于原文件使用新版 PyTorch ZIP 格式，PyTorch 1.2 无法直接读取，因此本项目在本机生成了兼容版本：

```text
pretrained_models/msdm_final_v3_legacy.pth
```

转换文件不会覆盖原权重。`train.py` 和两个测试脚本默认使用该兼容权重。

## 训练

激活环境后运行：

```powershell
conda activate msdm
python train.py
```

常用参数：

```powershell
python train.py `
  --data-dir data\data-used-by-train-val-test `
  --batch-size 10 `
  --max-epoch 200 `
  --log-interval 1 `
  --device 0
```

训练输出：

- 检查点与训练日志：`ckpts/`
- TensorBoard：`runs/`
- 运行指标：`log/run_log.txt`

程序会自动创建 `ckpts`、`runs` 和 `log` 目录。

训练会逐 batch 输出：

```text
Epoch 0 Batch 3/16 (18.8%), Loss: ..., WF: ..., FF: ...,
45.2s/batch, ETA: ..., GPU memory: ... GB
```

### 为什么训练很慢

每个 batch 会对两个物种分别计算 OT 损失，默认每次执行 100 轮 Sinkhorn 迭代：

```text
--num-of-iter-in-ot 100
```

因此 MS-DM 本身计算量较大。若 GPU 利用率持续接近 100%，通常表示仍在计算而不是卡死。调试时可以临时降低 OT 迭代次数或 batch size，但这会改变训练设置和结果。

## 统一测试与预测入口

建议在支持 RTX 4060 的现代 PyTorch 环境中运行：

```powershell
conda activate bisenet
python test.py
```

`test.py` 默认一次完成：

- 白粉虱和果蝇计数；
- 两类密度图保存；
- 两类位置文本保存；
- 带预测框的可视化图片；
- 每张图片的 `summary.csv` 汇总。

默认路径：

```text
输入：test_images-pre-result-full/data-used-by-train-val-test
统一输出：test_images-pre-result-full/result
权重：pretrained_models/msdm_final_v3_legacy.pth
```

显式指定参数：

```powershell
python test.py `
  --model-path pretrained_models\msdm_final_v3_legacy.pth `
  --data-path test_images-pre-result-full\data-used-by-train-val-test `
  --output-dir test_images-pre-result-full\result `
  --mode both `
  --device cuda
```

快速验证时可限制每个子集的图片数量：

```powershell
python test.py --subsets test --max-images 1
```

`--max-images 0` 表示处理全部图片。

运行模式：

```powershell
python test.py --mode count   # 计数和密度图
python test.py --mode points  # 位置、可视化和密度图
python test.py --mode both    # 全部输出，默认模式
```

统一输出目录结构：

```text
result/
├── density/whitefly/
├── density/fruit-fly/
├── positions/whitefly/
├── positions/fruit-fly/
├── visualizations/
└── summary.csv
```

## 图片尺寸与拼图

现有测试图片可能达到 1440×1920。脚本当前会将整张图片送入网络，因此全量测试的显存占用和运行时间明显高于 512×512 的训练裁剪。

拼图逻辑仅对文件名末尾包含两位行列编号的切片执行，例如：

```text
image_00.jpg
image_01.jpg
image_10.jpg
image_11.jpg
```

普通文件名（如 `0001.jpg`）不会再被错误地当作切片坐标。

## 已验证状态

- 最终 MS-DM 网络可导入；
- 兼容权重可严格加载，所有参数匹配；
- 双分支前向计算成功；
- 离散密度图保持计数守恒；
- 统一入口 `test.py` 已完成计数、密度图、位置文本、可视化和 CSV 的单图端到端验证；
- 全部 44 张大图尚未完成一次连续全量测试。

## 常见问题

### `FileNotFoundError: ./log/run_log.txt`

已修复。训练初始化时会自动创建 `log` 和 `runs`。

### `RuntimeError: ... is a zip archive`

PyTorch 1.2 无法读取新版 ZIP 权重。使用：

```text
pretrained_models/msdm_final_v3_legacy.pth
```

### `ModuleNotFoundError: No module named 'cv2'`

安装 OpenCV：

```powershell
pip install opencv-python
```

### RTX 4060 上长时间没有输出

先运行 `nvidia-smi` 检查 GPU。如果使用 PyTorch 1.2/CUDA 10，建议切换到支持 Ada 架构的 PyTorch 2.x/CUDA 12.x 环境。

### `PILLOW_VERSION` 导入错误

这是 TorchVision 0.4 与新版 Pillow 的兼容问题。仅在复现旧环境时使用兼容版本：

```powershell
pip install pillow==6.2.2
```

## 引用

```bibtex
@article{zhang2024msdm,
  title={A multi-species pest recognition and counting method based on a density map in the greenhouse},
  author={Zhang, Zhiqin and Rong, Jiacheng and Qi, Zhongxian and Yang, Yan and Zheng, Xiajun and Gao, Jin and Li, Wei and Yuan, Ting},
  journal={Computers and Electronics in Agriculture},
  volume={217},
  pages={108554},
  year={2024},
  issn={0168-1699},
  doi={10.1016/j.compag.2023.108554}
}
```

DM-Count：

```bibtex
@inproceedings{wang2020dmcount,
  title={Distribution Matching for Crowd Counting},
  author={Wang, Boyu and Liu, Huidong and Samaras, Dimitris and Nguyen, Minh Hoai},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

参考文献格式：

> Zhang, Z., Rong, J., Qi, Z., Yang, Y., Zheng, X., Gao, J., Li, W., & Yuan, T. (2024). A multi-species pest recognition and counting method based on a density map in the greenhouse. *Computers and Electronics in Agriculture, 217*, 108554. https://doi.org/10.1016/j.compag.2023.108554

> Wang, B., Liu, H., Samaras, D., & Nguyen, M. H. (2020). Distribution Matching for Crowd Counting. In *Advances in Neural Information Processing Systems* (Vol. 33).

## License

见 [LICENSE](LICENSE)。
